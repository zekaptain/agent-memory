package main

import (
	"image/color"

	"charm.land/bubbles/v2/help"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

// Until the terminal reports its background, styles use the terminal's own
// foreground. NO_COLOR also disables emphasis escapes, not just RGB colors.
type palette struct {
	noColor, known, dark bool
}

func (p palette) accent() color.Color {
	if p.noColor || !p.known {
		return nil
	}
	return lipgloss.LightDark(p.dark)(lipgloss.Color("#285D75"), lipgloss.Color("#87BBC9"))
}

func (p palette) muted() color.Color {
	if p.noColor || !p.known {
		return nil
	}
	return lipgloss.LightDark(p.dark)(lipgloss.Color("#5D6570"), lipgloss.Color("#9BA4B0"))
}

func (p palette) edge() color.Color {
	if p.noColor || !p.known {
		return nil
	}
	return lipgloss.LightDark(p.dark)(lipgloss.Color("#AAB5BF"), lipgloss.Color("#465361"))
}

func (p palette) title() lipgloss.Style {
	s := lipgloss.NewStyle().Bold(!p.noColor)
	if c := p.accent(); c != nil {
		s = s.Foreground(c)
	}
	return s
}

func (p palette) subtle() lipgloss.Style {
	s := lipgloss.NewStyle()
	if c := p.muted(); c != nil {
		s = s.Foreground(c)
	}
	return s
}

func (p palette) panel(width int) lipgloss.Style {
	s := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1).Width(width)
	if c := p.edge(); c != nil {
		s = s.BorderForeground(c)
	}
	return s
}

func (p palette) huhTheme() *huh.Styles {
	t := huh.ThemeBase(false)
	// Panels own borders and padding; embedded fields add neither.
	f := huh.FieldStyles{
		Base:           lipgloss.NewStyle(),
		Title:          p.title(),
		Description:    p.subtle(),
		SelectSelector: p.title().SetString("› "),
		Option:         lipgloss.NewStyle(),
		SelectedOption: p.title(),
		ErrorIndicator: lipgloss.NewStyle().SetString("! "),
		ErrorMessage:   p.subtle(),
		TextInput: huh.TextInputStyles{
			Cursor:      p.title(),
			Placeholder: p.subtle(),
			Prompt:      p.title(),
			Text:        lipgloss.NewStyle(),
		},
	}
	t.Focused, t.Blurred = f, f
	t.Form.Base, t.Group.Base = lipgloss.NewStyle(), lipgloss.NewStyle()
	t.Help = p.helpStyles()
	return t
}

func (p palette) helpStyles() help.Styles {
	return help.Styles{
		ShortKey: p.title(), ShortDesc: p.subtle(), ShortSeparator: p.subtle(),
		FullKey: p.title(), FullDesc: p.subtle(), FullSeparator: p.subtle(),
		Ellipsis: p.subtle(),
	}
}
