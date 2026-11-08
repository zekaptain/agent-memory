package main

import (
	"bytes"
	"image/color"
	"os"
	"regexp"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/x/ansi"
)

func TestNoColorUsesEnvironmentPresenceIncludingEmpty(t *testing.T) {
	t.Setenv("NO_COLOR", "")
	if !noColorRequested() {
		t.Fatal("NO_COLOR with empty value must disable colors")
	}
	t.Setenv("NO_COLOR", "0")
	if !noColorRequested() {
		t.Fatal("NO_COLOR=0 still disables colors")
	}
	if err := os.Unsetenv("NO_COLOR"); err != nil {
		t.Fatal(err)
	}
	if noColorRequested() {
		t.Fatal("absent NO_COLOR disables colors")
	}
}

func TestThemeWaitsForBackgroundAndSharesPaletteWithHuh(t *testing.T) {
	m, _ := testModel(t, syntheticScreen())
	m.styles.noColor = false
	if m.styles.accent() != nil || m.styles.dark {
		t.Fatal("frontend assumed dark background before detection")
	}
	field := m.selection
	press(m, "down")
	m.Update(tea.BackgroundColorMsg{Color: color.White})
	light := m.styles.huhTheme().Focused.SelectedOption.GetForeground()
	if light != m.styles.accent() || m.styles.dark {
		t.Fatal("Huh and Lip Gloss did not share detected light palette")
	}
	m.Update(tea.BackgroundColorMsg{Color: color.Black})
	dark := m.styles.huhTheme().Focused.SelectedOption.GetForeground()
	if dark != m.styles.accent() || dark == light || !m.styles.dark {
		t.Fatal("background message did not update shared dark palette")
	}
	if m.selection != field || m.selected != "no" {
		t.Fatal("theme change lost highlighted option")
	}
}

func TestLimitedColorProfilesPreserveLayoutAndDownsample(t *testing.T) {
	m, _ := testModel(t, compareScreen())
	m.styles.noColor = false
	m.Update(tea.BackgroundColorMsg{Color: color.Black})
	view := m.View().Content
	for _, profile := range []colorprofile.Profile{colorprofile.Ascii, colorprofile.ANSI, colorprofile.ANSI256} {
		var output bytes.Buffer
		writer := &colorprofile.Writer{Forward: &output, Profile: profile}
		if _, err := writer.WriteString(view); err != nil {
			t.Fatal(err)
		}
		got := output.String()
		if strings.Contains(got, "38;2;") || strings.Contains(got, "48;2;") {
			t.Fatalf("%s profile retained truecolor", profile)
		}
		if profile == colorprofile.ANSI &&
			(strings.Contains(got, "38;5;") || strings.Contains(got, "48;5;")) {
			t.Fatal("16-color profile retained 256-color escapes")
		}
		if profile == colorprofile.Ascii && regexp.MustCompile(`\x1b\[[0-9;]*(3[0-9]|4[0-9]|9[0-9]|10[0-7])m`).MatchString(got) {
			t.Fatal("ASCII profile retained color")
		}
		if ansi.Strip(got) != ansi.Strip(view) || lipgloss.Width(got) != lipgloss.Width(view) ||
			lipgloss.Height(got) != lipgloss.Height(view) {
			t.Fatal("color downsampling changed content or layout")
		}
	}
}
