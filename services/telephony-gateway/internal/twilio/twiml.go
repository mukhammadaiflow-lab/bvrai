package twilio

import (
	"encoding/xml"
	"fmt"
	"strings"
)

// TwiML response structures

// Response is the root TwiML element
type Response struct {
	XMLName xml.Name `xml:"Response"`
	Content []interface{}
}

// Connect TwiML verb for connecting media streams
type Connect struct {
	XMLName xml.Name `xml:"Connect"`
	Stream  *Stream  `xml:",omitempty"`
}

// Stream TwiML element for bidirectional media streaming
type Stream struct {
	XMLName              xml.Name `xml:"Stream"`
	URL                  string   `xml:"url,attr"`
	Name                 string   `xml:"name,attr,omitempty"`
	Track                string   `xml:"track,attr,omitempty"`
	StatusCallback       string   `xml:"statusCallback,attr,omitempty"`
	StatusCallbackMethod string   `xml:"statusCallbackMethod,attr,omitempty"`
	Parameters           []Parameter
}

// Parameter for passing custom parameters to the stream
type Parameter struct {
	XMLName xml.Name `xml:"Parameter"`
	Name    string   `xml:"name,attr"`
	Value   string   `xml:"value,attr"`
}

// Say TwiML verb for text-to-speech
type Say struct {
	XMLName  xml.Name `xml:"Say"`
	Voice    string   `xml:"voice,attr,omitempty"`
	Language string   `xml:"language,attr,omitempty"`
	Loop     int      `xml:"loop,attr,omitempty"`
	Text     string   `xml:",chardata"`
}

// Play TwiML verb for playing audio
type Play struct {
	XMLName xml.Name `xml:"Play"`
	Loop    int      `xml:"loop,attr,omitempty"`
	Digits  string   `xml:"digits,attr,omitempty"`
	URL     string   `xml:",chardata"`
}

// Pause TwiML verb for pausing
type Pause struct {
	XMLName xml.Name `xml:"Pause"`
	Length  int      `xml:"length,attr,omitempty"`
}

// Gather TwiML verb for collecting input
type Gather struct {
	XMLName         xml.Name `xml:"Gather"`
	Input           string   `xml:"input,attr,omitempty"`
	Action          string   `xml:"action,attr,omitempty"`
	Method          string   `xml:"method,attr,omitempty"`
	Timeout         int      `xml:"timeout,attr,omitempty"`
	SpeechTimeout   string   `xml:"speechTimeout,attr,omitempty"`
	Language        string   `xml:"language,attr,omitempty"`
	Hints           string   `xml:"hints,attr,omitempty"`
	PartialResultCallback string `xml:"partialResultCallback,attr,omitempty"`
	Content         []interface{}
}

// Hangup TwiML verb for ending call
type Hangup struct {
	XMLName xml.Name `xml:"Hangup"`
}

// Reject TwiML verb for rejecting call
type Reject struct {
	XMLName xml.Name `xml:"Reject"`
	Reason  string   `xml:"reason,attr,omitempty"`
}

// Dial TwiML verb for dialing
type Dial struct {
	XMLName        xml.Name `xml:"Dial"`
	Action         string   `xml:"action,attr,omitempty"`
	Method         string   `xml:"method,attr,omitempty"`
	Timeout        int      `xml:"timeout,attr,omitempty"`
	HangupOnStar   bool     `xml:"hangupOnStar,attr,omitempty"`
	TimeLimit      int      `xml:"timeLimit,attr,omitempty"`
	CallerID       string   `xml:"callerId,attr,omitempty"`
	Record         string   `xml:"record,attr,omitempty"`
	RecordingStatusCallback string `xml:"recordingStatusCallback,attr,omitempty"`
	Number         string   `xml:",chardata"`
}

// Redirect TwiML verb for redirecting
type Redirect struct {
	XMLName xml.Name `xml:"Redirect"`
	Method  string   `xml:"method,attr,omitempty"`
	URL     string   `xml:",chardata"`
}

// StreamConfig holds configuration for stream TwiML
type StreamConfig struct {
	URL                  string
	Track                string   // inbound_track, outbound_track, both_tracks
	StatusCallback       string
	StatusCallbackEvents []string
	Parameters           map[string]string
}

// GenerateStreamTwiML generates TwiML for bidirectional streaming
func GenerateStreamTwiML(cfg StreamConfig) string {
	var params []Parameter
	for name, value := range cfg.Parameters {
		params = append(params, Parameter{Name: name, Value: value})
	}

	track := cfg.Track
	if track == "" {
		track = "both_tracks"
	}

	stream := &Stream{
		URL:                  cfg.URL,
		Track:                track,
		StatusCallback:       cfg.StatusCallback,
		StatusCallbackMethod: "POST",
		Parameters:           params,
	}

	response := Response{
		Content: []interface{}{
			Connect{Stream: stream},
		},
	}

	return marshalTwiML(response)
}

// GenerateSayTwiML generates TwiML for text-to-speech
func GenerateSayTwiML(text, voice, language string) string {
	if voice == "" {
		voice = "Polly.Amy"
	}
	if language == "" {
		language = "en-US"
	}

	response := Response{
		Content: []interface{}{
			Say{Text: text, Voice: voice, Language: language},
		},
	}

	return marshalTwiML(response)
}

// GenerateSayAndStreamTwiML generates TwiML that says something then connects stream
func GenerateSayAndStreamTwiML(greeting string, streamCfg StreamConfig) string {
	var params []Parameter
	for name, value := range streamCfg.Parameters {
		params = append(params, Parameter{Name: name, Value: value})
	}

	track := streamCfg.Track
	if track == "" {
		track = "both_tracks"
	}

	stream := &Stream{
		URL:                  streamCfg.URL,
		Track:                track,
		StatusCallback:       streamCfg.StatusCallback,
		StatusCallbackMethod: "POST",
		Parameters:           params,
	}

	response := Response{
		Content: []interface{}{
			Say{Text: greeting, Voice: "Polly.Amy", Language: "en-US"},
			Connect{Stream: stream},
		},
	}

	return marshalTwiML(response)
}

// GenerateGatherTwiML generates TwiML for gathering speech input
func GenerateGatherTwiML(prompt string, actionURL string, timeout int) string {
	response := Response{
		Content: []interface{}{
			Gather{
				Input:         "speech",
				Action:        actionURL,
				Method:        "POST",
				Timeout:       timeout,
				SpeechTimeout: "auto",
				Language:      "en-US",
				Content: []interface{}{
					Say{Text: prompt, Voice: "Polly.Amy", Language: "en-US"},
				},
			},
		},
	}

	return marshalTwiML(response)
}

// GenerateHangupTwiML generates TwiML to hang up
func GenerateHangupTwiML() string {
	response := Response{
		Content: []interface{}{
			Hangup{},
		},
	}

	return marshalTwiML(response)
}

// GenerateRejectTwiML generates TwiML to reject a call
func GenerateRejectTwiML(reason string) string {
	if reason == "" {
		reason = "rejected"
	}
	response := Response{
		Content: []interface{}{
			Reject{Reason: reason},
		},
	}

	return marshalTwiML(response)
}

// GenerateTransferTwiML generates TwiML to transfer to another number
func GenerateTransferTwiML(number, callerID string, recordingCallback string) string {
	dial := Dial{
		Number:   number,
		Timeout:  30,
		CallerID: callerID,
	}

	if recordingCallback != "" {
		dial.Record = "record-from-answer-dual"
		dial.RecordingStatusCallback = recordingCallback
	}

	response := Response{
		Content: []interface{}{
			dial,
		},
	}

	return marshalTwiML(response)
}

// GenerateHoldTwiML generates TwiML for placing caller on hold
func GenerateHoldTwiML(holdMusicURL string) string {
	response := Response{
		Content: []interface{}{
			Play{URL: holdMusicURL, Loop: 0}, // Loop forever
		},
	}

	return marshalTwiML(response)
}

// marshalTwiML marshals the response to XML string
func marshalTwiML(response Response) string {
	var builder strings.Builder
	builder.WriteString(`<?xml version="1.0" encoding="UTF-8"?>`)

	data, err := xml.Marshal(response)
	if err != nil {
		return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?><Response><Say>An error occurred</Say></Response>`)
	}

	builder.Write(data)
	return builder.String()
}
