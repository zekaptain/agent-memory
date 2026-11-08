package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"image/color"
	"io"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func testModel(t *testing.T, s screen) (*model, *bytes.Buffer) {
	t.Helper()
	var out bytes.Buffer
	m := newModel(bufio.NewReader(strings.NewReader("")), &out, true)
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 32})
	deliverScreen(t, m, s, 1)
	return m, &out
}

func deliverScreen(t *testing.T, m *model, s screen, id int64) tea.Cmd {
	t.Helper()
	f, err := decodeFrame(screenWire(t, s, id))
	if err != nil {
		t.Fatal(err)
	}
	_, cmd := m.Update(frameMsg{frame: f})
	if m.err != nil {
		t.Fatal(m.err)
	}
	return cmd
}

func keyMsg(name string) tea.KeyPressMsg {
	codes := map[string]rune{
		"up": tea.KeyUp, "down": tea.KeyDown, "left": tea.KeyLeft, "right": tea.KeyRight,
		"enter": tea.KeyEnter, "esc": tea.KeyEscape, "pgup": tea.KeyPgUp, "pgdown": tea.KeyPgDown,
		"home": tea.KeyHome, "end": tea.KeyEnd, "tab": tea.KeyTab, "space": tea.KeySpace,
	}
	if code, ok := codes[name]; ok {
		return tea.KeyPressMsg{Code: code}
	}
	if name == "ctrl+c" {
		return tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl}
	}
	return tea.KeyPressMsg{Code: []rune(name)[0], Text: name}
}

func press(m *model, name string) tea.Cmd {
	_, cmd := m.Update(keyMsg(name))
	return cmd
}

func sentAction(t *testing.T, m *model, out *bytes.Buffer, cmd tea.Cmd) action {
	t.Helper()
	if cmd == nil {
		t.Fatal("expected action command")
	}
	msg := cmd()
	if _, ok := msg.(sentMsg); !ok {
		t.Fatalf("not an action command: %T", msg)
	}
	m.Update(msg)
	var got action
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	return got
}

func TestSelectionUsesOnlyArrowsAndEnter(t *testing.T) {
	m, out := testModel(t, syntheticScreen())
	if m.selected != "yes" {
		t.Fatal("first option should start highlighted")
	}
	for _, letter := range []string{"j", "k", "h", "l", "/", "n", "y", "B", "g", "G", "home", "end", "left", "right", "tab", "space"} {
		if cmd := press(m, letter); cmd != nil || m.selected != "yes" || m.waiting {
			t.Fatalf("%q changed selection or submitted", letter)
		}
	}
	press(m, "down")
	if m.selected != "no" {
		t.Fatal("down did not move Huh selection")
	}
	if !strings.Contains(ansi.Strip(m.View().Content), "› No") {
		t.Fatal("down changed bound value but did not redraw highlight")
	}
	press(m, "down")
	press(m, "up")
	if m.selected != "no" {
		t.Fatal("up did not move Huh selection")
	}
	a := sentAction(t, m, out, press(m, "enter"))
	if a.Action != "answer" || a.Value != "no" || a.RequestID != 1 {
		t.Fatalf("wrong highlighted answer: %+v", a)
	}
}

func TestEvidencePagingNeverSubmitsAndKeepsQuestionFixed(t *testing.T) {
	m, out := testModel(t, syntheticScreen())
	before := m.View().Content
	questionLine := strings.Index(before, "Did the turn fail?")
	press(m, "down")
	press(m, "pgdown")
	if m.evidence.YOffset() == 0 || m.selected != "no" || m.waiting || out.Len() != 0 {
		t.Fatal("evidence paging altered form/submitted")
	}
	after := m.View().Content
	if strings.Count(before[:questionLine], "\n") !=
		strings.Count(after[:strings.Index(after, "Did the turn fail?")], "\n") {
		t.Fatal("evidence scrolling moved fixed question")
	}
	press(m, "pgup")
	if m.evidence.YOffset() != 0 {
		t.Fatal("page up did not restore evidence")
	}
}

func TestPagersRestoreHighlightAndEvidenceOffset(t *testing.T) {
	m, out := testModel(t, syntheticScreen())
	press(m, "down")
	press(m, "pgdown")
	offset := m.evidence.YOffset()
	for _, shortcut := range []string{"p", "?"} {
		press(m, shortcut)
		if m.pagerTitle == "" {
			t.Fatal("pager not opened")
		}
		press(m, "pgdown")
		press(m, "down")
		if m.pager.YOffset() == 0 {
			t.Fatal("pager did not scroll")
		}
		pagerOffset := m.pager.YOffset()
		press(m, "enter")
		press(m, "s")
		press(m, "j")
		if m.waiting || m.pager.YOffset() != pagerOffset {
			t.Fatal("pager accepted a disabled key or submitted")
		}
		press(m, "end")
		if !m.pager.AtBottom() {
			t.Fatal("end did not reach bottom")
		}
		press(m, "home")
		if !m.pager.AtTop() {
			t.Fatal("home did not reach top")
		}
		press(m, "q")
		if m.pagerTitle != "" || m.selected != "no" ||
			m.evidence.YOffset() != offset || out.Len() != 0 {
			t.Fatal("pager return lost state or sent action")
		}
	}
	press(m, "p")
	press(m, "esc")
	if m.pagerTitle != "" {
		t.Fatal("escape did not close pager")
	}
}

func compareScreen() screen {
	s := syntheticScreen()
	s.Subject, s.Key, s.Question = "Compare drafts", "1:preferred", "Which lesson is better?"
	s.Options = []option{{"A", "Draft A"}, {"B", "Draft B"}, {"tie", "Tie"}}
	s.Drafts = []draft{
		{"A", "Lesson Alpha [literal]", strings.Repeat("Alpha cited evidence 界\n", 50)},
		{"B", "Lesson Beta", strings.Repeat("Beta cited evidence e\u0301\n", 50)},
	}
	s.Evidence = ""
	return s
}

func TestBackOnComparisonIsNotDraftBAndRestoresControllerAnswer(t *testing.T) {
	s := compareScreen()
	s.Selected = "B"
	m, out := testModel(t, s)
	press(m, "pgdown")
	offset := m.evidence.YOffset()
	a := sentAction(t, m, out, press(m, "b"))
	if a.Action != "back" || a.Value != "" {
		t.Fatalf("b must be a navigation action, not draft B: %+v", a)
	}
	deliverScreen(t, m, s, 2)
	if m.selected != "B" || m.evidence.YOffset() != offset {
		t.Fatal("controller selection/evidence offset not restored on back")
	}
	out.Reset()
	a = sentAction(t, m, out, press(m, "enter"))
	if a.Action != "answer" || a.Value != "B" || a.RequestID != 2 {
		t.Fatal("enter did not choose the restored draft B")
	}
}

func notesScreen() screen {
	s := syntheticScreen()
	s.Subject, s.Key, s.Question = "Notes", "1:notes", "Add a note?"
	s.Note = "Existing "
	s.Options = []option{{"edit", "Add/edit note"}, {"continue", "Continue"}}
	return s
}

func TestNotesUseBoundHuhInputAndLiteralShortcuts(t *testing.T) {
	m, out := testModel(t, notesScreen())
	press(m, "enter") // edit is strictly local.
	if !m.editing || m.waiting || out.Len() != 0 || m.note != "Existing " {
		t.Fatal("edit sent an action or failed to bind existing note")
	}
	for _, letter := range []string{"q", "s", "b", "r", "p", "?", "j", "/", "界"} {
		press(m, letter)
	}
	if m.note != "Existing qsbrp?j/界" || m.pagerTitle != "" || m.waiting {
		t.Fatalf("shortcuts were not literal note text: %q", m.note)
	}
	a := sentAction(t, m, out, press(m, "enter"))
	if a.Action != "note" || a.Value != "Existing qsbrp?j/界" {
		t.Fatalf("wrong bound note: %+v", a)
	}
	s := notesScreen()
	s.Note, s.Selected = a.Value, "continue"
	deliverScreen(t, m, s, 2)
	if m.editing || m.selected != "continue" {
		t.Fatal("accepted note did not return to selected Continue")
	}
	out.Reset()
	a = sentAction(t, m, out, press(m, "enter"))
	if a.Action != "answer" || a.Value != "continue" {
		t.Fatal("Continue did not reach controller")
	}
}

func TestNoteCancelAndSanitizedPaste(t *testing.T) {
	m, out := testModel(t, notesScreen())
	press(m, "enter")
	m.Update(tea.PasteMsg{Content: "\x1b[31m pasted\nline\x1b]52;c;bad\a"})
	if m.note != "Existing  pasted line" {
		t.Fatalf("unsafe note paste: %q", m.note)
	}
	press(m, "esc")
	if m.editing || m.screen.Note != "Existing " || out.Len() != 0 || m.waiting {
		t.Fatal("cancel saved or failed to exit editor")
	}
	press(m, "enter")
	if m.note != "Existing " {
		t.Fatal("cancelled edit replaced controller note")
	}
	m.Update(tea.PasteMsg{Content: strings.Repeat("界", 1400)})
	if cmd := press(m, "enter"); cmd != nil || m.waiting || m.noteError == "" {
		t.Fatal("multibyte note exceeded byte limit without validation")
	}
}

func TestWaitingSaveRequiresControllerAcknowledgment(t *testing.T) {
	s := syntheticScreen()
	s.Subject, s.Question = "Summary", "Save these labels?"
	s.Options = []option{{"save", "Save labels"}, {"redo", "Redo item"}}
	s.CanRedo = true
	m, out := testModel(t, s)
	cmd := press(m, "enter")
	if !m.waiting {
		t.Fatal("must enter waiting before sending action")
	}
	before := out.Len()
	for _, name := range []string{"enter", "down", "s", "b", "r"} {
		if press(m, name) != nil || out.Len() != before {
			t.Fatal("a second action escaped while waiting")
		}
	}
	a := sentAction(t, m, out, cmd)
	if a.Action != "answer" || a.Value != "save" || !m.waiting {
		t.Fatal("saving did not wait for controller")
	}
	v := ansi.Strip(m.View().Content)
	if !strings.Contains(v, "Waiting for controller") || strings.Contains(v, "Saved") {
		t.Fatal("frontend claimed local save success")
	}
	_, done := m.Update(frameMsg{frame: frame{1, 2, "done", nil, "Saved"}})
	if done == nil || m.err != nil {
		t.Fatal("done acknowledgment did not quit cleanly")
	}
	if _, ok := done().(tea.QuitMsg); !ok {
		t.Fatal("done did not use normal Tea cleanup")
	}
}

func TestNavigationAvailabilityAndDistinctRedo(t *testing.T) {
	for _, subject := range []string{"Turn", "Review", "Summary"} {
		s := syntheticScreen()
		s.Subject, s.CanBack, s.CanRedo = subject, false, true
		m, out := testModel(t, s)
		if press(m, "b") != nil || m.waiting {
			t.Fatal("disabled back emitted")
		}
		cmd := press(m, "r")
		if subject == "Turn" {
			if cmd != nil || m.waiting {
				t.Fatal("redo available outside review/summary")
			}
		} else if a := sentAction(t, m, out, cmd); a.Action != "redo" {
			t.Fatal("redo shortcut conflated with answer")
		}
	}
	for _, shortcut := range []string{"s", "q", "ctrl+c"} {
		m, out := testModel(t, syntheticScreen())
		a := sentAction(t, m, out, press(m, shortcut))
		want := "quit"
		if shortcut == "s" {
			want = "skip"
		}
		if a.Action != want {
			t.Fatalf("%s emitted %s", shortcut, a.Action)
		}
	}
}

func TestModelRejectsRequestReorderingAndUnsolicitedScreens(t *testing.T) {
	for _, tc := range []struct {
		id      int64
		waiting bool
	}{{1, false}, {0, true}, {1, true}, {2, false}} {
		m, _ := testModel(t, syntheticScreen())
		m.waiting = tc.waiting
		f := frame{1, tc.id, "screen", m.screen, ""}
		_, cmd := m.Update(frameMsg{frame: f})
		if m.err != errProtocol || cmd == nil {
			t.Fatalf("invalid request accepted: %+v", tc)
		}
	}
}

func TestErrorsAndEOFUseNormalTeaQuit(t *testing.T) {
	for _, e := range []error{errClosed, errProtocol, errWrite} {
		m, _ := testModel(t, syntheticScreen())
		var msg tea.Msg = frameMsg{err: e}
		if e == errWrite {
			msg = sentMsg{err: e}
		}
		_, cmd := m.Update(msg)
		if m.err != e {
			t.Fatal("error not propagated")
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Fatal("error bypassed normal Tea cleanup")
		}
	}
}

func TestControllerEOFExitsRealProgramWhileHumanIdleOrWaiting(t *testing.T) {
	for _, waiting := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "waiting"}[waiting], func(t *testing.T) {
			pr, pw := io.Pipe()
			defer pr.Close()
			defer pw.Close()
			var out bytes.Buffer
			m := newModel(bufio.NewReader(pr), &out, true)
			ready := make(chan struct{}, 1)
			sent := make(chan struct{}, 1)
			p := tea.NewProgram(m, tea.WithInput(nil), tea.WithOutput(io.Discard), tea.WithoutRenderer(),
				tea.WithWindowSize(100, 32), tea.WithFilter(func(_ tea.Model, msg tea.Msg) tea.Msg {
					switch v := msg.(type) {
					case frameMsg:
						if v.frame.Type == "screen" {
							ready <- struct{}{}
						}
					case sentMsg:
						sent <- struct{}{}
					}
					return msg
				}))
			result := make(chan error, 1)
			go func() {
				_, err := p.Run()
				result <- err
			}()
			if _, err := pw.Write(screenWire(t, syntheticScreen(), 1)); err != nil {
				t.Fatal(err)
			}
			select {
			case <-ready:
			case <-time.After(3 * time.Second):
				p.Kill()
				t.Fatal("program did not receive screen")
			}
			if waiting {
				p.Send(keyMsg("enter"))
				select {
				case <-sent:
				case <-time.After(3 * time.Second):
					p.Kill()
					t.Fatal("program did not send action")
				}
			}
			pw.Close()
			select {
			case err := <-result:
				if err != nil || m.err != errClosed {
					t.Fatalf("EOF not normal cleanup: run=%v model=%v", err, m.err)
				}
			case <-time.After(3 * time.Second):
				p.Kill()
				t.Fatal("EOF did not stop program while user was idle")
			}
		})
	}
}

func TestResizePreservesFormAndScrollableEvidence(t *testing.T) {
	m, _ := testModel(t, syntheticScreen())
	press(m, "down")
	press(m, "pgdown")
	field := m.selection
	offset := m.evidence.YOffset()
	for _, size := range [][2]int{{80, 30}, {140, 50}, {40, 35}, {100, 32}} {
		m.Update(tea.WindowSizeMsg{Width: size[0], Height: size[1]})
		if m.selected != "no" || m.selection != field || m.evidence.YOffset() != offset {
			t.Fatalf("resize lost state at %v: selected=%s offset=%d", size, m.selected, m.evidence.YOffset())
		}
		if m.tooSmall {
			t.Fatalf("usable size %v considered too small", size)
		}
		assertFits(t, m, size[0], size[1])
	}
}

func TestWidthChangesRestoreTerminalTabStops(t *testing.T) {
	m, _ := testModel(t, syntheticScreen())
	for _, width := range []int{60, 42, 110, 60, 140} {
		_, cmd := m.Update(tea.WindowSizeMsg{Width: width, Height: 38})
		if cmd == nil {
			t.Fatal("width change did not restore tab stops and repaint")
		}
	}
	_, cmd := m.Update(tea.WindowSizeMsg{Width: 140, Height: 45})
	if cmd != nil {
		t.Fatal("height-only resize unnecessarily reset tab stops")
	}
}

func assertFits(t *testing.T, m *model, width, height int) {
	t.Helper()
	v := m.View()
	if !v.AltScreen {
		t.Fatal("frontend left full-screen mode")
	}
	if lipgloss.Width(v.Content) > width || lipgloss.Height(v.Content) > height {
		t.Fatalf("render %dx%d exceeds %dx%d:\n%s", lipgloss.Width(v.Content),
			lipgloss.Height(v.Content), width, height, v.Content)
	}
}

func TestUnicodeLayoutsAndLightDarkNoColor(t *testing.T) {
	for _, s := range []screen{syntheticScreen(), compareScreen(), notesScreen()} {
		for _, noColor := range []bool{false, true} {
			for _, bg := range []color.Color{color.Black, color.White} {
				for _, size := range [][2]int{{140, 45}, {80, 30}, {40, 35}, {20, 10}, {100, 9}, {1, 1}} {
					m, _ := testModel(t, s)
					m.styles.noColor = noColor
					m.Update(tea.BackgroundColorMsg{Color: bg})
					m.Update(tea.WindowSizeMsg{Width: size[0], Height: size[1]})
					assertFits(t, m, size[0], size[1])
					view := m.View().Content
					if noColor && strings.Contains(view, "\x1b") {
						t.Fatal("NO_COLOR view contains ANSI styling")
					}
					if size[0] < 38 || size[1] < 15 {
						if !m.tooSmall || !strings.Contains(view, "Resize") && size[0] > 10 {
							t.Fatal("undersized controls were not replaced by resize message")
						}
					}
				}
			}
		}
	}
}

func TestComparisonColumnsStackAtNarrowWidths(t *testing.T) {
	m, _ := testModel(t, compareScreen())
	m.Update(tea.WindowSizeMsg{Width: 140, Height: 40})
	content := ansi.Strip(m.evidenceContent(m.evidence.Width()))
	a, b := strings.Index(content, "Draft A · Lesson"), strings.Index(content, "Draft B · Lesson")
	if a < 0 || b < 0 || strings.Count(content[:a], "\n") != strings.Count(content[:b], "\n") {
		t.Fatal("wide comparison did not put drafts side by side")
	}
	if !strings.Contains(content, "Alpha cited evidence 界") || !strings.Contains(content, "[literal]") {
		t.Fatal("comparison lost literal lesson or citation")
	}
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})
	content = ansi.Strip(m.evidenceContent(m.evidence.Width()))
	a, b = strings.Index(content, "Draft A · Lesson"), strings.Index(content, "Draft B · Lesson")
	if strings.Count(content[:a], "\n") >= strings.Count(content[:b], "\n") {
		t.Fatal("narrow comparison did not stack drafts")
	}
	press(m, "end") // Not a selection jump.
	if m.selected != "A" {
		t.Fatal("disabled letter/jump navigation returned")
	}
}

func TestEditorAndPagerResizeRetainState(t *testing.T) {
	m, _ := testModel(t, notesScreen())
	press(m, "enter")
	press(m, "界")
	editor := m.editor
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 35})
	if m.editor != editor || m.note != "Existing 界" || !m.editing {
		t.Fatal("resize reconstructed editor")
	}
	assertFits(t, m, 80, 35)
	press(m, "esc")
	press(m, "down")
	press(m, "pgdown")
	evidenceOffset := m.evidence.YOffset()
	press(m, "p")
	press(m, "pgdown")
	pagerOffset := m.pager.YOffset()
	m.Update(tea.WindowSizeMsg{Width: 70, Height: 35})
	if m.pager.YOffset() != pagerOffset {
		t.Fatal("resize lost pager offset")
	}
	assertFits(t, m, 70, 35)
	press(m, "esc")
	if m.selected != "continue" || m.evidence.YOffset() != evidenceOffset {
		t.Fatal("pager resize/return lost form or evidence")
	}
}

func TestTooSmallAllowsQuitWithoutInvisibleAnswer(t *testing.T) {
	m, out := testModel(t, syntheticScreen())
	m.Update(tea.WindowSizeMsg{Width: 30, Height: 8})
	if !m.tooSmall || press(m, "enter") != nil || m.waiting {
		t.Fatal("hidden controls accepted an answer")
	}
	if a := sentAction(t, m, out, press(m, "q")); a.Action != "quit" {
		t.Fatal("too-small view disabled quit")
	}
}

func TestUnicodeQuestionAndOptionWrapUseDisplayCells(t *testing.T) {
	s := syntheticScreen()
	s.Question = strings.Repeat("界 e\u0301 👩‍💻 ", 8) + "[literal question]"
	s.Options = []option{
		{"yes", strings.Repeat("界 e\u0301 ", 8) + "Yes"},
		{"no", strings.Repeat("👩‍💻 ", 8) + "No"},
	}
	m, _ := testModel(t, s)
	for _, width := range []int{40, 80, 140} {
		m.Update(tea.WindowSizeMsg{Width: width, Height: 60})
		if m.tooSmall {
			t.Fatal("Unicode controls should fit this tall screen")
		}
		assertFits(t, m, width, 60)
		view := ansi.Strip(m.View().Content)
		if !strings.Contains(view, "[literal") || !strings.Contains(view, "question]") {
			t.Fatal("literal brackets were interpreted or dropped")
		}
	}
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 12})
	if !m.tooSmall || press(m, "enter") != nil {
		t.Fatal("wrapped controls submitted while not visible")
	}
}

func TestFooterContainsOnlyAvailableControlsInWrappedPairs(t *testing.T) {
	s := syntheticScreen()
	s.CanBack = false
	m, _ := testModel(t, s)
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 40})
	footer := ansi.Strip(m.footer)
	if strings.Contains(footer, "b back") || strings.Contains(footer, "r redo") ||
		!strings.Contains(footer, "enter confirm") || !strings.Contains(footer, "p projection") ||
		!strings.Contains(footer, "? guide") || !strings.Contains(footer, "s skip") ||
		!strings.Contains(footer, "q quit") || !strings.Contains(footer, "\n") {
		t.Fatalf("wrong available/wrapped key help: %s", footer)
	}
	if lipgloss.Width(footer) > 38 {
		t.Fatal("footer exceeded measured available width")
	}
}
