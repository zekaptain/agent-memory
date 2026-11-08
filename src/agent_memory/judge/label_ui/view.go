package main

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func wrap(s string, width int) string {
	return ansi.Wrap(s, max(1, width), "")
}

// Footer pairs come from the same active Bubbles bindings that handle input.
// Wrap whole pairs instead of letting help's single line silently omit keys.
func (m *model) renderFooter(width int) string {
	h := help.New()
	h.Styles = m.styles.helpStyles()
	h.SetWidth(width)
	var lines []string
	line := ""
	for _, binding := range m.activeBindings() {
		pair := h.ShortHelpView([]key.Binding{binding})
		if pair == "" {
			continue
		}
		next := pair
		if line != "" {
			next = line + "   " + pair
		}
		if line != "" && lipgloss.Width(next) > width {
			lines = append(lines, line)
			line = pair
		} else {
			line = next
		}
	}
	if line != "" {
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func (m *model) renderQuestion(width int) string {
	if m.screen == nil {
		return ""
	}
	inner := max(1, width-m.styles.panel(width).GetHorizontalFrameSize())
	m.selection.WithWidth(inner)
	form := m.selection.View()
	if m.editing && m.editor != nil {
		m.editor.WithWidth(inner)
		form = nonempty(m.styles.subtle().Render("Note · single line · 4096 bytes maximum"),
			m.editor.View(), m.styles.subtle().Render(m.noteError))
	}
	status := ""
	if m.waiting {
		status = m.styles.subtle().Render("Waiting for controller…")
	}
	body := nonempty(m.styles.title().Render(wrap(m.screen.Question, inner)),
		form, status, m.styles.subtle().Render(wrap(m.screen.Notice, inner)))
	return m.styles.panel(width).Render(body)
}

func (m *model) draftCard(d draft, width int) string {
	inner := max(1, width-m.styles.panel(width).GetHorizontalFrameSize())
	body := nonempty(
		m.styles.title().Render("Draft "+d.Slot+" · Lesson"),
		wrap(d.Lesson, inner),
		"",
		m.styles.subtle().Render("\nCited evidence"),
		wrap(d.CitedEvidence, inner),
	)
	return m.styles.panel(width).Render(body)
}

func (m *model) evidenceContent(width int) string {
	s := m.screen
	if len(s.Drafts) == 0 {
		return wrap(nonempty(s.Evidence, notesText(s)), width)
	}
	var cards []string
	wide := len(s.Drafts) == 2 && width >= 100
	cardWidth := width
	if wide {
		cardWidth = (width - 1) / 2
	}
	for _, d := range s.Drafts {
		cards = append(cards, m.draftCard(d, cardWidth))
	}
	var content string
	if wide {
		content = lipgloss.JoinHorizontal(lipgloss.Top, cards[0], " ", cards[1])
	} else {
		content = strings.Join(cards, "\n")
	}
	return nonempty(content, wrap(s.Evidence, width), wrap(notesText(s), width))
}

func notesText(s *screen) string {
	if s.Subject == "Notes" && s.Note != "" {
		return "Current note\n" + s.Note
	}
	return ""
}

// layout changes component dimensions/content, never reconstructs fields.
// Evidence and pager offsets are retained, then clamped by the viewport.
func (m *model) layout() {
	if m.width <= 0 || m.height <= 0 || m.screen == nil {
		return
	}
	width := max(1, m.width-2)
	inner := max(1, width-m.styles.panel(width).GetHorizontalFrameSize())
	m.header = nonempty(
		m.styles.title().Render(ansi.Truncate(fmt.Sprintf("JUDGE / LABEL   %d of %d   ·   %s",
			m.screen.Item, m.screen.Total, m.screen.Subject), width, "…")),
		m.styles.subtle().Render(ansi.Truncate(cleanLine(m.screen.Metadata), width, "…")),
	)
	m.footer = m.renderFooter(width)
	m.question = m.renderQuestion(width)
	offset := m.evidence.YOffset()
	m.evidence.SetWidth(inner)
	// Notes can contain a large projection too. Typing or a waiting-state
	// change must not rewrap megabytes of unchanged evidence per keystroke.
	if m.contentScreen != m.screen || m.contentWidth != inner || m.contentPalette != m.styles {
		m.evidence.SetContent(m.evidenceContent(inner))
		m.contentScreen, m.contentWidth, m.contentPalette = m.screen, inner, m.styles
	}
	// Evidence panel has two border lines and a one-line section title.
	evidenceHeight := m.height - lipgloss.Height(m.header) -
		lipgloss.Height(m.question) - lipgloss.Height(m.footer) - 3
	m.tooSmall = m.width < 38 || evidenceHeight < 2
	if !m.tooSmall {
		m.evidence.SetHeight(evidenceHeight)
		m.evidence.SetYOffset(offset)
	}
	if m.pagerTitle != "" {
		pagerOffset := m.pager.YOffset()
		m.pager.SetWidth(inner)
		m.pager.SetContent(wrap(m.pagerText, inner))
		pagerHeight := m.height - lipgloss.Height(m.header) - lipgloss.Height(m.footer) - 3
		m.tooSmall = m.width < 38 || pagerHeight < 2
		if !m.tooSmall {
			m.pager.SetHeight(pagerHeight)
			m.pager.SetYOffset(pagerOffset)
		}
	}
}

func (m *model) View() (v tea.View) {
	defer func() {
		if recover() != nil {
			v = tea.NewView("Terminal interface unavailable. Press ctrl+c to quit.")
			v.AltScreen = true
		}
	}()
	v.AltScreen = true
	width := max(1, m.width-2)
	if m.screen == nil {
		v.SetContent(wrap("Connecting to labeling controller…\nq quit", width))
		return v
	}
	if m.tooSmall {
		msg := "Resize terminal to show evidence and controls.\nctrl+c quits"
		if !m.editing && m.pagerTitle == "" {
			msg = "Resize terminal to show evidence and controls.\nq quits"
		}
		v.SetContent(lipgloss.NewStyle().MaxHeight(max(1, m.height)).Render(wrap(msg, max(1, m.width))))
		return v
	}
	var body string
	if m.pagerTitle != "" {
		tag := m.styles.subtle().Render(fmt.Sprintf("%s   ·   %d%%", m.pagerTitle,
			int(m.pager.ScrollPercent()*100)))
		body = m.styles.panel(width).Render(nonempty(tag, m.pager.View()))
	} else {
		tag := "Evidence"
		if len(m.screen.Drafts) > 0 {
			tag = m.screen.Subject
		}
		tag = m.styles.subtle().Render(fmt.Sprintf("%s   ·   %d%%", tag,
			int(m.evidence.ScrollPercent()*100)))
		// Render the live Huh field, not layout's height-measurement snapshot:
		// arrows and input-cursor ticks do not require an evidence reflow.
		body = nonempty(m.styles.panel(width).Render(tag+"\n"+m.evidence.View()), m.renderQuestion(width))
	}
	content := nonempty(m.header, body, m.footer)
	v.SetContent(lipgloss.NewStyle().Padding(0, 1).Render(content))
	return v
}
