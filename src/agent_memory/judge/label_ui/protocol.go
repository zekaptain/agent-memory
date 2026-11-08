package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/charmbracelet/x/ansi"
)

const (
	protocolVersion = 1
	maxFrame        = 8 * 1024 * 1024 // Includes the terminating newline.
	maxNote         = 4096
)

var (
	errProtocol = errors.New("invalid labeling controller message")
	errClosed   = errors.New("labeling controller disconnected")
	errWrite    = errors.New("cannot send labeling action")
)

type option struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

type draft struct {
	Slot          string `json:"slot"`
	Lesson        string `json:"lesson"`
	CitedEvidence string `json:"cited_evidence"`
}

type screen struct {
	Item       int      `json:"item"`
	Total      int      `json:"total"`
	Subject    string   `json:"subject"`
	Key        string   `json:"key"`
	Metadata   string   `json:"metadata"`
	Evidence   string   `json:"evidence"`
	Projection string   `json:"projection"`
	Guide      string   `json:"guide"`
	Drafts     []draft  `json:"drafts"`
	Question   string   `json:"question"`
	Options    []option `json:"options"`
	Selected   string   `json:"selected"`
	Note       string   `json:"note"`
	CanBack    bool     `json:"can_back"`
	CanRedo    bool     `json:"can_redo"`
	Notice     string   `json:"notice"`
}

type frame struct {
	Version   int     `json:"version"`
	RequestID int64   `json:"request_id"`
	Type      string  `json:"type"`
	Screen    *screen `json:"screen,omitempty"`
	Message   string  `json:"message,omitempty"`
}

type action struct {
	Version   int    `json:"version"`
	RequestID int64  `json:"request_id"`
	Type      string `json:"type"`
	Action    string `json:"action"`
	Value     string `json:"value"`
}

// readFrame bounds allocation even for a peer that never sends a newline.
// A partial final line is not a frame: NDJSON always requires its delimiter.
func readFrame(r *bufio.Reader) (frame, error) {
	var data []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(data)+len(part) > maxFrame {
			return frame{}, errProtocol
		}
		data = append(data, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil {
			if err == io.EOF && len(data) == 0 {
				return frame{}, errClosed
			}
			return frame{}, errProtocol
		}
		return decodeFrame(data)
	}
}

// readJSON rejects duplicate keys as well as unknown/missing keys. Ordinary
// encoding/json struct decoding silently accepts duplicates and null values.
func readJSON(d *json.Decoder, depth int) (any, error) {
	if depth > 8 {
		return nil, errProtocol
	}
	t, err := d.Token()
	if err != nil {
		return nil, errProtocol
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return t, nil
	}
	switch delim {
	case '{':
		obj := make(map[string]any)
		for d.More() {
			k, err := d.Token()
			name, ok := k.(string)
			if err != nil || !ok {
				return nil, errProtocol
			}
			if _, exists := obj[name]; exists {
				return nil, errProtocol
			}
			v, err := readJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
			obj[name] = v
		}
		end, err := d.Token()
		if err != nil || end != json.Delim('}') {
			return nil, errProtocol
		}
		return obj, nil
	case '[':
		var list []any
		for d.More() {
			v, err := readJSON(d, depth+1)
			if err != nil {
				return nil, err
			}
			list = append(list, v)
		}
		end, err := d.Token()
		if err != nil || end != json.Delim(']') {
			return nil, errProtocol
		}
		return list, nil
	}
	return nil, errProtocol
}

func shape(v any, stringsRequired, intsRequired, boolsRequired, arraysRequired string) bool {
	obj, ok := v.(map[string]any)
	if !ok {
		return false
	}
	fields := 0
	for _, group := range []struct{ names, kind string }{
		{stringsRequired, "string"}, {intsRequired, "int"},
		{boolsRequired, "bool"}, {arraysRequired, "array"},
	} {
		for _, name := range strings.Fields(group.names) {
			fields++
			switch group.kind {
			case "string":
				if _, ok := obj[name].(string); !ok {
					return false
				}
			case "int":
				n, ok := obj[name].(json.Number)
				if !ok {
					return false
				}
				if _, err := n.Int64(); err != nil {
					return false
				}
			case "bool":
				if _, ok := obj[name].(bool); !ok {
					return false
				}
			case "array":
				if _, ok := obj[name].([]any); !ok {
					return false
				}
			}
		}
	}
	return len(obj) == fields
}

func decodeFrame(data []byte) (frame, error) {
	if len(data) == 0 || len(data) > maxFrame || data[len(data)-1] != '\n' ||
		!utf8.Valid(data) || bytes.Contains(data[:len(data)-1], []byte{'\n'}) {
		return frame{}, errProtocol
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := readJSON(d, 0)
	if err != nil {
		return frame{}, errProtocol
	}
	if _, err := d.Token(); err != io.EOF {
		return frame{}, errProtocol
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return frame{}, errProtocol
	}
	switch obj["type"] {
	case "screen":
		base := map[string]any{"version": obj["version"], "request_id": obj["request_id"], "type": obj["type"]}
		if len(obj) != 4 || !shape(base, "type", "version request_id", "", "") ||
			!shape(obj["screen"], "subject key metadata evidence projection guide question selected note notice",
				"item total", "can_back can_redo", "drafts options") {
			return frame{}, errProtocol
		}
		s := obj["screen"].(map[string]any)
		for _, item := range s["drafts"].([]any) {
			if !shape(item, "slot lesson cited_evidence", "", "", "") {
				return frame{}, errProtocol
			}
		}
		for _, item := range s["options"].([]any) {
			if !shape(item, "id label", "", "", "") {
				return frame{}, errProtocol
			}
		}
	case "done":
		if !shape(obj, "type message", "version request_id", "", "") {
			return frame{}, errProtocol
		}
	default:
		return frame{}, errProtocol
	}
	var f frame
	if err := json.Unmarshal(data, &f); err != nil || f.Version != protocolVersion || f.RequestID <= 0 {
		return frame{}, errProtocol
	}
	if f.Type == "done" {
		f.Message = clean(f.Message)
		return f, nil
	}
	s := f.Screen
	if s.Item < 0 || s.Total < s.Item || s.Key == "" || cleanLine(s.Key) != s.Key ||
		len(s.Options) == 0 || len(s.Options) > 64 || len(s.Drafts) > 2 ||
		len(s.Note) > maxNote || cleanLine(s.Note) != s.Note {
		return frame{}, errProtocol
	}
	switch s.Subject {
	case "Turn", "Draft A", "Draft B", "Compare drafts", "Notes", "Review", "Summary":
	default:
		return frame{}, errProtocol
	}
	seen := make(map[string]bool)
	selectedFound := s.Selected == ""
	for i := range s.Options {
		o := &s.Options[i]
		if o.ID == "" || len(o.ID) > 128 || seen[o.ID] || cleanLine(o.ID) != o.ID {
			return frame{}, errProtocol
		}
		seen[o.ID] = true
		selectedFound = selectedFound || s.Selected == o.ID
		o.Label = clean(o.Label)
	}
	if !selectedFound {
		return frame{}, errProtocol
	}
	seen = make(map[string]bool)
	for i := range s.Drafts {
		d := &s.Drafts[i]
		if (d.Slot != "A" && d.Slot != "B") || seen[d.Slot] {
			return frame{}, errProtocol
		}
		seen[d.Slot] = true
		d.Lesson, d.CitedEvidence = clean(d.Lesson), clean(d.CitedEvidence)
	}
	s.Metadata, s.Evidence, s.Projection = clean(s.Metadata), clean(s.Evidence), clean(s.Projection)
	s.Guide, s.Question, s.Notice = clean(s.Guide), clean(s.Question), clean(s.Notice)
	return f, nil
}

// clean treats all controller content as literal text, never Rich/ANSI markup.
// Strip complete terminal sequences before removing residual control runes.
func clean(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = ansi.Strip(s)
	s = strings.Map(func(r rune) rune {
		if r == '\n' || r == '\t' {
			return r
		}
		if unicode.IsControl(r) || (unicode.Is(unicode.Cf, r) && r != '\u200d' && r != '\u200c') {
			return -1
		}
		return r
	}, s)
	return strings.ReplaceAll(s, "\t", "    ")
}

func cleanLine(s string) string {
	return strings.ReplaceAll(clean(s), "\n", " ")
}

func encodeAction(a action) ([]byte, error) {
	if a.Version != protocolVersion || a.RequestID <= 0 || a.Type != "action" {
		return nil, errProtocol
	}
	switch a.Action {
	case "answer":
		if a.Value == "" || cleanLine(a.Value) != a.Value {
			return nil, errProtocol
		}
	case "note":
		if len(a.Value) > maxNote || cleanLine(a.Value) != a.Value {
			return nil, errProtocol
		}
	case "back", "skip", "quit", "redo":
		if a.Value != "" {
			return nil, errProtocol
		}
	default:
		return nil, errProtocol
	}
	data, err := json.Marshal(a)
	if err != nil || len(data)+1 > maxFrame {
		return nil, errProtocol
	}
	return append(data, '\n'), nil
}

func writeAction(w io.Writer, a action) error {
	data, err := encodeAction(a)
	if err != nil {
		return err
	}
	n, err := w.Write(data)
	if err != nil || n != len(data) {
		return errWrite
	}
	return nil
}
