package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func syntheticScreen() screen {
	return screen{
		Item: 1, Total: 3, Subject: "Turn", Key: "1:failed",
		Metadata:   "project synthetic · closed by completion",
		Evidence:   strings.Repeat("[1] user: synthetic evidence 界 e\u0301 👩‍💻\n", 100),
		Projection: strings.Repeat("[1] Full projection [literal brackets]\n", 130),
		Guide:      strings.Repeat("Grounded: supported by the evidence.\n", 100),
		Drafts:     []draft{},
		Question:   "Did the turn fail?",
		Options:    []option{{"yes", "Yes"}, {"no", "No"}, {"unclear", "Unclear"}},
		CanBack:    true,
	}
}

func wireFrame(t *testing.T, f frame) []byte {
	t.Helper()
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	return append(data, '\n')
}

func screenWire(t *testing.T, s screen, id int64) []byte {
	t.Helper()
	return wireFrame(t, frame{Version: protocolVersion, RequestID: id, Type: "screen", Screen: &s})
}

func multiDraftScreen(count int) screen {
	s := syntheticScreen()
	s.Subject, s.Key, s.Question = "Compare drafts", "1:preferred", "Which draft would you keep?"
	s.Evidence, s.Options = "", nil
	for i := 0; i < count; i++ {
		slot := string(rune('A' + i))
		s.Drafts = append(s.Drafts, draft{slot, "Lesson " + slot + " [literal] 界",
			strings.Repeat("Cited evidence for "+slot+"\n", 8), ""})
		s.Options = append(s.Options, option{strings.ToLower(slot), "Draft " + slot})
	}
	s.Options = append(s.Options, option{"same", "All equally"}, option{"neither", "None"})
	s.Selected = strings.ToLower(s.Drafts[count-1].Slot)
	return s
}

func TestProtocolThreeFourDrafts(t *testing.T) {
	for _, count := range []int{3, 4} {
		s := multiDraftScreen(count)
		for _, subject := range []string{"Compare drafts", "Notes", "Review", "Summary"} {
			s.Subject = subject
			f, err := decodeFrame(screenWire(t, s, 1))
			if err != nil || len(f.Screen.Drafts) != count {
				t.Fatalf("%s with %d drafts rejected: %v", subject, count, err)
			}
		}
		for _, d := range s.Drafts {
			focused := s
			focused.Subject, focused.Drafts = "Draft "+d.Slot, []draft{d}
			focused.Drafts[0].Focus = "Claim 2 of 2"
			if _, err := decodeFrame(screenWire(t, focused, 1)); err != nil {
				t.Fatalf("focused %s rejected: %v", d.Slot, err)
			}
		}
	}
}

func TestProtocolRoundTrip(t *testing.T) {
	s := syntheticScreen()
	data := screenWire(t, s, 42)
	f, err := readFrame(bufio.NewReaderSize(bytes.NewReader(data), 17))
	if err != nil || f.RequestID != 42 || f.Screen.Key != s.Key ||
		f.Screen.Evidence != s.Evidence || len(f.Screen.Options) != 3 {
		t.Fatalf("round trip: frame=%+v error=%v", f, err)
	}
	for _, subject := range []string{"Turn", "Draft A", "Draft B", "Compare drafts", "Notes", "Review", "Summary"} {
		s.Subject = subject
		if _, err := decodeFrame(screenWire(t, s, 1)); err != nil {
			t.Fatalf("subject %s: %v", subject, err)
		}
	}
	done, err := decodeFrame([]byte("{\"version\":2,\"request_id\":43,\"type\":\"done\",\"message\":\"Finished\"}\n"))
	if err != nil || done.Message != "Finished" {
		t.Fatal("done frame rejected")
	}
}

func TestProtocolRejectsMalformedFrames(t *testing.T) {
	valid := string(screenWire(t, syntheticScreen(), 1))
	cases := map[string]string{
		"unknown root field":  strings.Replace(valid, `"version":2`, `"unknown":"private sentinel","version":2`, 1),
		"unknown screen":      strings.Replace(valid, `"key":`, `"secret":"private sentinel","key":`, 1),
		"unknown option":      strings.Replace(valid, `"id":"yes"`, `"id":"yes","secret":"private sentinel"`, 1),
		"duplicate root":      strings.Replace(valid, `"version":2`, `"version":2,"version":2`, 1),
		"duplicate nested":    strings.Replace(valid, `"id":"yes"`, `"id":"yes","id":"no"`, 1),
		"missing field":       strings.Replace(valid, `"selected":"",`, "", 1),
		"null string":         strings.Replace(valid, `"selected":""`, `"selected":null`, 1),
		"null bool":           strings.Replace(valid, `"can_back":true`, `"can_back":null`, 1),
		"null array":          strings.Replace(valid, `"drafts":[]`, `"drafts":null`, 1),
		"fractional id":       strings.Replace(valid, `"request_id":1`, `"request_id":1.1`, 1),
		"string id":           strings.Replace(valid, `"request_id":1`, `"request_id":"1"`, 1),
		"zero id":             strings.Replace(valid, `"request_id":1`, `"request_id":0`, 1),
		"negative id":         strings.Replace(valid, `"request_id":1`, `"request_id":-1`, 1),
		"overflow id":         strings.Replace(valid, `"request_id":1`, `"request_id":9223372036854775808`, 1),
		"wrong version":       strings.Replace(valid, `"version":2`, `"version":1`, 1),
		"unknown selected":    strings.Replace(valid, `"selected":""`, `"selected":"absent"`, 1),
		"bad subject":         strings.Replace(valid, `"subject":"Turn"`, `"subject":"private sentinel"`, 1),
		"two objects":         valid[:len(valid)-1] + "{}\n",
		"physical newline":    "{}\n{}\n",
		"missing newline":     valid[:len(valid)-1],
		"not an object":       "[]\n",
		"invalid JSON":        "{\"private sentinel\":\n",
		"invalid UTF8":        "{\"version\":\"\xff\"}\n",
		"unexpected done key": "{\"version\":2,\"request_id\":2,\"type\":\"done\",\"message\":\"\",\"screen\":{}}\n",
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := decodeFrame([]byte(data))
			if !errors.Is(err, errProtocol) || strings.Contains(err.Error(), "private sentinel") {
				t.Fatalf("expected safe protocol error, got %v", err)
			}
		})
	}
}

func TestProtocolRequiresDraftFocus(t *testing.T) {
	s := syntheticScreen()
	s.Subject = "Draft A"
	s.Drafts = []draft{{"A", "Rule: Check first.", "[2] evidence", "Claim 2 of 3"}}
	valid := screenWire(t, s, 1)
	f, err := decodeFrame(valid)
	if err != nil || f.Screen.Drafts[0].Focus != "Claim 2 of 3" {
		t.Fatalf("focus rejected: %v", err)
	}
	missing := strings.Replace(string(valid), `,"focus":"Claim 2 of 3"`, "", 1)
	if _, err := decodeFrame([]byte(missing)); err != errProtocol {
		t.Fatalf("missing focus accepted: %v", err)
	}
	s.Drafts[0].Focus = ""
	if _, err := decodeFrame(screenWire(t, s, 1)); err != nil {
		t.Fatalf("empty draft-level focus rejected: %v", err)
	}
}

func TestProtocolValidatesScreenSemantics(t *testing.T) {
	tests := map[string]func(*screen){
		"negative item":    func(s *screen) { s.Item = -1 },
		"invalid total":    func(s *screen) { s.Total = 0 },
		"empty key":        func(s *screen) { s.Key = "" },
		"control key":      func(s *screen) { s.Key = "1:\x1b[2Jfailed" },
		"no options":       func(s *screen) { s.Options = []option{} },
		"duplicate option": func(s *screen) { s.Options = append(s.Options, s.Options[0]) },
		"empty option":     func(s *screen) { s.Options[0].ID = "" },
		"control option":   func(s *screen) { s.Options[0].ID = "yes\nno" },
		"long note":        func(s *screen) { s.Note = strings.Repeat("x", maxNote+1) },
		"multiline note":   func(s *screen) { s.Note = "line\nline" },
		"hidden identity":  func(s *screen) { s.Drafts = []draft{{Slot: "model-name"}} },
		"duplicate slot":   func(s *screen) { s.Drafts = []draft{{Slot: "A"}, {Slot: "A"}} },
		"empty slot":       func(s *screen) { s.Drafts = []draft{{Slot: ""}} },
		"substring slot":   func(s *screen) { s.Drafts = []draft{{Slot: "AB"}} },
		"lowercase slot":   func(s *screen) { s.Drafts = []draft{{Slot: "a"}} },
		"fifth slot":       func(s *screen) { s.Drafts = []draft{{Slot: "E"}} },
		"fifth subject":    func(s *screen) { s.Subject = "Draft E" },
		"too many drafts":  func(s *screen) { *s = multiDraftScreen(5) },
	}
	for name, edit := range tests {
		t.Run(name, func(t *testing.T) {
			s := syntheticScreen()
			edit(&s)
			if _, err := decodeFrame(screenWire(t, s, 1)); err != errProtocol {
				t.Fatalf("expected protocol error, got %v", err)
			}
		})
	}
}

func TestFrameBoundIncludesNewline(t *testing.T) {
	base := []byte("{\"version\":2,\"request_id\":1,\"type\":\"done\",\"message\":\"\"}\n")
	exact := bytes.Replace(base, []byte(`"message":""`),
		[]byte(`"message":"`+strings.Repeat("x", maxFrame-len(base))+`"`), 1)
	if len(exact) != maxFrame {
		t.Fatal("test did not construct exact frame")
	}
	if _, err := readFrame(bufio.NewReader(bytes.NewReader(exact))); err != nil {
		t.Fatalf("exact bound rejected: %v", err)
	}
	over := append(append([]byte{}, exact[:len(exact)-1]...), ' ', '\n')
	if _, err := readFrame(bufio.NewReader(bytes.NewReader(over))); err != errProtocol {
		t.Fatalf("oversized frame accepted: %v", err)
	}
	if _, err := readFrame(bufio.NewReader(strings.NewReader(strings.Repeat("x", maxFrame+1)))); err != errProtocol {
		t.Fatalf("unbounded non-delimited input accepted: %v", err)
	}
}

func TestEOFAndTruncatedFrameAreDistinct(t *testing.T) {
	if _, err := readFrame(bufio.NewReader(strings.NewReader(""))); err != errClosed {
		t.Fatalf("EOF: %v", err)
	}
	if _, err := readFrame(bufio.NewReader(strings.NewReader("{}"))); err != errProtocol {
		t.Fatalf("truncated frame: %v", err)
	}
	r := bufio.NewReader(bytes.NewReader(append(
		wireFrame(t, frame{protocolVersion, 1, "done", nil, "One"}),
		wireFrame(t, frame{protocolVersion, 2, "done", nil, "Two"})...)))
	for _, id := range []int64{1, 2} {
		f, err := readFrame(r)
		if err != nil || f.RequestID != id {
			t.Fatal("frame boundaries were lost")
		}
	}
	if _, err := readFrame(r); err != errClosed {
		t.Fatal("expected EOF after two frames")
	}
}

func TestSanitizeTerminalControlsButKeepLiteralText(t *testing.T) {
	original := "[red]界 👩‍💻 e\u0301[/red]\x1b[31m red\x1b[0m\x1b]52;c;c2VjcmV0\a\x1b[2J\x00\x7f\u009b\u202e\n\tend"
	got := clean(original)
	if got != "[red]界 👩‍💻 e\u0301[/red] red\n    end" {
		t.Fatalf("sanitization: %q", got)
	}
	s := syntheticScreen()
	s.Question = original
	s.Options[0].Label = "[blue]Yes[/blue]\x1b[31m"
	s.Drafts = []draft{{"A", original, original, original}}
	f, err := decodeFrame(screenWire(t, s, 1))
	if err != nil || f.Screen.Question != got || f.Screen.Drafts[0].Lesson != got ||
		f.Screen.Options[0].Label != "[blue]Yes[/blue]" || f.Screen.Drafts[0].Focus != got {
		t.Fatal("display fields were not sanitized")
	}
}

func TestActionValidationAndBounds(t *testing.T) {
	for _, name := range []string{"answer", "back", "skip", "quit", "redo", "note"} {
		value := ""
		if name == "answer" {
			value = "B"
		}
		if name == "note" {
			value = strings.Repeat("界", maxNote/3) + "x"
		}
		a := action{protocolVersion, 9, "action", name, value}
		var buf bytes.Buffer
		if err := writeAction(&buf, a); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var got action
		if err := json.Unmarshal(buf.Bytes(), &got); err != nil || got != a ||
			buf.Bytes()[buf.Len()-1] != '\n' {
			t.Fatal("action frame does not match")
		}
	}
	for _, a := range []action{
		{1, 1, "action", "skip", ""}, {protocolVersion, 0, "action", "skip", ""},
		{protocolVersion, 1, "screen", "skip", ""}, {protocolVersion, 1, "action", "bad", ""},
		{protocolVersion, 1, "action", "back", "B"}, {protocolVersion, 1, "action", "note", "a\nb"},
		{protocolVersion, 1, "action", "note", strings.Repeat("界", maxNote/3+1)},
		{protocolVersion, 1, "action", "answer", strings.Repeat("x", maxFrame)},
	} {
		if _, err := encodeAction(a); err != errProtocol {
			t.Fatalf("invalid action accepted: %v", err)
		}
	}
	if err := writeAction(shortWriter{}, action{protocolVersion, 1, "action", "quit", ""}); err != errWrite {
		t.Fatalf("short write not detected: %v", err)
	}
	if err := writeAction(errorWriter{}, action{protocolVersion, 1, "action", "quit", ""}); err != errWrite {
		t.Fatalf("write error not safe: %v", err)
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

type errorWriter struct{}

func (errorWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
