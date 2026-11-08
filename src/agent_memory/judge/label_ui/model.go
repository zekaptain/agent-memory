package main

import (
	"bufio"
	"errors"
	"io"
	"strings"

	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"github.com/charmbracelet/x/ansi"
)

var errUI = errors.New("terminal interface could not continue")

type frameMsg struct {
	frame frame
	err   error
}

type sentMsg struct{ err error }

type bindings struct {
	choose, enter, pages, back, skip, quit, redo, projection, guide key.Binding
	pagerMove, pagerHome, pagerEnd, close, cancel                   key.Binding
}

func bind(keys []string, display, description string) key.Binding {
	return key.NewBinding(key.WithKeys(keys...), key.WithHelp(display, description))
}

func newBindings() bindings {
	return bindings{
		choose:     bind([]string{"up", "down"}, "↑/↓", "choose"),
		enter:      bind([]string{"enter"}, "enter", "confirm"),
		pages:      bind([]string{"pgup", "pgdown"}, "pgup/pgdn", "evidence"),
		back:       bind([]string{"b"}, "b", "back"),
		skip:       bind([]string{"s"}, "s", "skip"),
		quit:       bind([]string{"q", "ctrl+c"}, "q", "quit"),
		redo:       bind([]string{"r"}, "r", "redo"),
		projection: bind([]string{"p"}, "p", "projection"),
		guide:      bind([]string{"?"}, "?", "guide"),
		pagerMove:  bind([]string{"up", "down", "pgup", "pgdown"}, "↑/↓ pgup/pgdn", "scroll"),
		pagerHome:  bind([]string{"home"}, "home", "start"),
		pagerEnd:   bind([]string{"end"}, "end", "end"),
		close:      bind([]string{"esc", "q"}, "esc/q", "return"),
		cancel:     bind([]string{"esc"}, "esc", "cancel"),
	}
}

type model struct {
	reader *bufio.Reader
	writer io.Writer
	keys   bindings
	styles palette

	width, height int
	requestID     int64
	screen        *screen
	selected      string // Huh is always bound to this frontend-owned value.
	selection     *huh.Select[string]
	editor        *huh.Input
	note          string
	noteError     string
	editing       bool
	waiting       bool
	err           error

	evidence   viewport.Model
	pager      viewport.Model
	pagerTitle string
	pagerText  string
	offsets    map[string]int
	tooSmall   bool
	header     string
	question   string
	footer     string

	contentScreen  *screen
	contentWidth   int
	contentPalette palette
}

func newViewport(pager bool) viewport.Model {
	v := viewport.New()
	v.FillHeight = true
	v.MouseWheelEnabled = false
	// No inherited vim keys, Enter/space paging, or letter shortcuts.
	v.KeyMap = viewport.KeyMap{
		PageUp:   bind([]string{"pgup"}, "pgup", "page up"),
		PageDown: bind([]string{"pgdown"}, "pgdn", "page down"),
	}
	if pager {
		v.KeyMap.Up = bind([]string{"up"}, "↑", "up")
		v.KeyMap.Down = bind([]string{"down"}, "↓", "down")
	}
	return v
}

func newModel(r *bufio.Reader, w io.Writer, noColor bool) *model {
	return &model{
		reader: r, writer: w, keys: newBindings(),
		styles:   palette{noColor: noColor},
		evidence: newViewport(false), pager: newViewport(true),
		offsets: make(map[string]int),
	}
}

func (m *model) readNext() tea.Cmd {
	return func() tea.Msg {
		f, err := readFrame(m.reader)
		return frameMsg{f, err}
	}
}

func (m *model) Init() tea.Cmd {
	// A read remains pending even while the human is considering a question.
	// EOF is an event, not something discovered only after the next answer.
	return tea.Batch(m.readNext(), tea.RequestBackgroundColor)
}

func (m *model) fail(err error) (tea.Model, tea.Cmd) {
	m.err = err
	return m, tea.Quit
}

func (m *model) Update(msg tea.Msg) (next tea.Model, cmd tea.Cmd) {
	// Never expose a panic value or traceback containing controller content.
	defer func() {
		if recover() != nil {
			next, cmd = m.fail(errUI)
		}
	}()
	return m.update(msg)
}

func (m *model) update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case frameMsg:
		if msg.err != nil {
			return m.fail(msg.err)
		}
		f := msg.frame
		if f.RequestID <= m.requestID || (m.screen != nil && !m.waiting && f.Type != "done") {
			return m.fail(errProtocol)
		}
		m.requestID = f.RequestID
		if f.Type == "done" {
			return m, tea.Quit
		}
		if m.screen != nil {
			m.offsets[m.screen.Key] = m.evidence.YOffset()
		}
		m.screen = f.Screen
		m.waiting, m.editing = false, false
		m.pagerTitle, m.pagerText, m.noteError = "", "", ""
		m.editor = nil
		m.selected = m.screen.Selected
		m.makeSelection()
		m.evidence.GotoTop()
		m.layout()
		m.evidence.SetYOffset(m.offsets[m.screen.Key])
		return m, m.readNext()
	case sentMsg:
		if msg.err != nil {
			return m.fail(msg.err)
		}
		return m, nil // Only a new controller frame acknowledges an action.
	case tea.WindowSizeMsg:
		widthChanged := m.width > 0 && msg.Width != m.width
		m.width, m.height = msg.Width, msg.Height
		m.layout()
		if widthChanged {
			// Bubble Tea #1838: terminals can lose hard-tab stops when narrowed.
			// Re-establish them before repainting after a width change, using
			// upstream's workaround until the pinned renderer handles it.
			return m, tea.Sequence(tea.Raw(ansi.SetTabEvery8Columns), tea.ClearScreen)
		}
		return m, nil
	case tea.BackgroundColorMsg:
		m.styles.known, m.styles.dark = true, msg.IsDark()
		// Themes share the palette through their closure, without rebuilding
		// either field or losing the highlighted option / input cursor.
		if m.selection != nil {
			_, _ = m.selection.Update(msg)
		}
		if m.editor != nil {
			_, _ = m.editor.Update(msg)
		}
		m.layout()
		return m, nil
	case tea.KeyPressMsg:
		return m.keypress(msg)
	case tea.PasteMsg:
		if !m.editing || m.waiting || m.tooSmall {
			return m, nil
		}
		msg.Content = cleanLine(msg.Content)
		_, cmd := m.editor.Update(msg)
		m.note = cleanLine(m.note)
		m.layout()
		return m, cmd
	}
	// Cursor blink and other Huh input messages are local to the editor.
	if m.editing && m.editor != nil {
		_, cmd := m.editor.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m *model) makeSelection() {
	options := make([]huh.Option[string], len(m.screen.Options))
	for i, o := range m.screen.Options {
		options[i] = huh.NewOption(o.Label, o.ID)
	}
	if m.selected == "" {
		m.selected = options[0].Value
	}
	m.selection = huh.NewSelect[string]().Options(options...).Value(&m.selected)
	// A fresh map, not NewDefaultKeyMap: no filter, tabs, letters, or jumps.
	km := &huh.KeyMap{Select: huh.SelectKeyMap{
		Up:     bind([]string{"up"}, "↑", "up"),
		Down:   bind([]string{"down"}, "↓", "down"),
		Submit: m.keys.enter,
	}}
	m.selection.WithKeyMap(km)
	m.selection.WithTheme(huh.ThemeFunc(func(bool) *huh.Styles { return m.styles.huhTheme() }))
	m.selection.Focus()
}

func (m *model) openEditor() tea.Cmd {
	m.editing, m.noteError = true, ""
	m.note = m.screen.Note
	m.editor = huh.NewInput().Value(&m.note).CharLimit(maxNote).
		Placeholder("A short, optional note").
		Validate(func(value string) error {
			if len(value) > maxNote || cleanLine(value) != value {
				return errors.New("Use a single line of at most 4096 bytes")
			}
			return nil
		})
	m.editor.WithKeyMap(&huh.KeyMap{Input: huh.InputKeyMap{Submit: m.keys.enter}})
	m.editor.WithTheme(huh.ThemeFunc(func(bool) *huh.Styles { return m.styles.huhTheme() }))
	cmd := m.editor.Focus()
	m.layout()
	return cmd
}

func (m *model) send(name, value string) (tea.Model, tea.Cmd) {
	if m.waiting || m.screen == nil {
		return m, nil
	}
	switch name {
	case "answer":
		found := false
		for _, o := range m.screen.Options {
			found = found || o.ID == value
		}
		if !found || (m.screen.Subject == "Notes" && value == "edit") {
			return m.fail(errProtocol)
		}
	case "back":
		if !m.screen.CanBack {
			return m, nil
		}
	case "redo":
		if !m.canRedo() {
			return m, nil
		}
	case "note":
		if !m.editing || m.screen.Subject != "Notes" {
			return m.fail(errProtocol)
		}
	}
	a := action{protocolVersion, m.requestID, "action", name, value}
	if _, err := encodeAction(a); err != nil {
		return m.fail(err)
	}
	m.waiting = true // Before writing: key repeats cannot send another action.
	m.layout()
	return m, func() tea.Msg { return sentMsg{writeAction(m.writer, a)} }
}

func (m *model) canRedo() bool {
	return m.screen != nil && m.screen.CanRedo &&
		(m.screen.Subject == "Summary" || m.screen.Subject == "Review")
}

func (m *model) keypress(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if m.waiting || m.screen == nil {
		if key.Matches(msg, m.keys.quit) {
			// Escape hatch while a controller is unavailable; never emit a
			// second action for the same request.
			return m, tea.Quit
		}
		return m, nil
	}
	if msg.String() == "ctrl+c" {
		return m.send("quit", "")
	}
	if m.tooSmall {
		if key.Matches(msg, m.keys.quit) && !m.editing && m.pagerTitle == "" {
			return m.send("quit", "")
		}
		if key.Matches(msg, m.keys.cancel) && m.editing {
			m.editing, m.editor = false, nil
			m.layout()
		}
		if key.Matches(msg, m.keys.close) && m.pagerTitle != "" {
			m.pagerTitle = ""
			m.layout()
		}
		return m, nil
	}
	if m.pagerTitle != "" {
		switch {
		case key.Matches(msg, m.keys.close):
			m.pagerTitle, m.pagerText = "", ""
			m.layout()
		case key.Matches(msg, m.keys.pagerHome):
			m.pager.GotoTop()
		case key.Matches(msg, m.keys.pagerEnd):
			m.pager.GotoBottom()
		case key.Matches(msg, m.keys.pagerMove):
			m.pager, _ = m.pager.Update(msg)
		}
		return m, nil
	}
	if m.editing {
		switch {
		case key.Matches(msg, m.keys.cancel):
			m.editing, m.editor = false, nil
			m.noteError = ""
			m.layout()
			return m, nil
		case key.Matches(msg, m.keys.enter):
			m.note = cleanLine(m.note)
			if len(m.note) > maxNote {
				m.noteError = "Use at most 4096 bytes."
				m.layout()
				return m, nil
			}
			return m.send("note", m.note)
		}
		// q/s/b/r/p/? are literal text here. Disallow clipboard commands:
		// bracketed paste is supported and sanitized without external IO.
		if msg.String() == "ctrl+v" || msg.String() == "alt+v" {
			return m, nil
		}
		if msg.Text != "" {
			msg.Text = cleanLine(msg.Text)
			if msg.Text == "" {
				return m, nil
			}
		}
		_, cmd := m.editor.Update(msg)
		m.noteError = ""
		m.layout()
		return m, cmd
	}
	switch {
	case key.Matches(msg, m.keys.quit):
		return m.send("quit", "")
	case key.Matches(msg, m.keys.skip):
		return m.send("skip", "")
	case key.Matches(msg, m.keys.back):
		return m.send("back", "")
	case key.Matches(msg, m.keys.redo):
		return m.send("redo", "")
	case key.Matches(msg, m.keys.projection):
		m.pagerTitle, m.pagerText = "Full projection", m.screen.Projection
		m.pager.GotoTop()
		m.layout()
	case key.Matches(msg, m.keys.guide):
		m.pagerTitle, m.pagerText = "Labeling guide", m.screen.Guide
		m.pager.GotoTop()
		m.layout()
	case key.Matches(msg, m.keys.pages):
		m.evidence, _ = m.evidence.Update(msg)
	case key.Matches(msg, m.keys.choose):
		_, _ = m.selection.Update(msg)
	case key.Matches(msg, m.keys.enter):
		// Let Huh commit the bound highlighted value. Its NextField command
		// is intentionally not run: the controller owns question navigation.
		_, _ = m.selection.Update(msg)
		if m.screen.Subject == "Notes" && m.selected == "edit" {
			return m, m.openEditor()
		}
		return m.send("answer", m.selected)
	}
	return m, nil
}

func (m *model) activeBindings() []key.Binding {
	if m.waiting || m.screen == nil {
		return []key.Binding{m.keys.quit}
	}
	if m.pagerTitle != "" {
		return []key.Binding{m.keys.pagerMove, m.keys.pagerHome, m.keys.pagerEnd, m.keys.close}
	}
	if m.editing {
		return []key.Binding{m.keys.enter, m.keys.cancel}
	}
	keys := []key.Binding{m.keys.choose, m.keys.enter, m.keys.pages, m.keys.projection, m.keys.guide}
	if m.screen.CanBack {
		keys = append(keys, m.keys.back)
	}
	if m.canRedo() {
		keys = append(keys, m.keys.redo)
	}
	return append(keys, m.keys.skip, m.keys.quit)
}

func nonempty(parts ...string) string {
	var kept []string
	for _, p := range parts {
		if strings.TrimSpace(p) != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, "\n")
}
