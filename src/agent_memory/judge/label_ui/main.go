package main

import (
	"bufio"
	"fmt"
	"os"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/colorprofile"
)

func main() {
	if err := run(); err != nil {
		// All errors returned by run are fixed, content-free diagnostics.
		fmt.Fprintln(os.Stderr, "Labeling frontend:", err)
		os.Exit(1)
	}
}

func run() error {
	// Bubble Tea's opt-in tracing records rendered evidence. This frontend
	// must never create such logs, even if inherited from a developer shell.
	_ = os.Unsetenv("TEA_TRACE")
	_ = os.Unsetenv("TEA_DEBUG")
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return fmt.Errorf("cannot open controlling terminal")
	}
	defer tty.Close()

	noColor := noColorRequested()
	m := newModel(bufio.NewReader(os.Stdin), os.Stdout, noColor)
	opts := []tea.ProgramOption{tea.WithInput(tty), tea.WithOutput(tty)}
	if noColor {
		opts = append(opts, tea.WithColorProfile(colorprofile.Ascii))
	}
	// No logging options, no stdout UI, no arguments or environment containing
	// controller data. Program.Run restores the terminal before returning.
	p := tea.NewProgram(m, opts...)
	final, err := p.Run()
	if err != nil {
		return fmt.Errorf("terminal session ended unexpectedly")
	}
	return final.(*model).err
}

func noColorRequested() bool {
	_, present := os.LookupEnv("NO_COLOR")
	return present // Presence, including NO_COLOR="".
}
