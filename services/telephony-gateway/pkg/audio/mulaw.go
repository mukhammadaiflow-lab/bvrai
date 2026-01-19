package audio

import (
	"math"
)

// μ-law companding constants
const (
	MulawMax   = 32635
	MulawBias  = 132
	MulawClip  = 32635

	// Sample rates
	SampleRate8K  = 8000
	SampleRate16K = 16000
	SampleRate24K = 24000
	SampleRate48K = 48000

	// Twilio uses 8kHz mulaw
	TwilioSampleRate = 8000
	TwilioFrameSize  = 160 // 20ms at 8kHz
)

// MulawEncodeTable is the encoding lookup table
var MulawEncodeTable = [256]int16{
	0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
}

// MulawDecodeTable is the decoding lookup table
var MulawDecodeTable = [256]int16{
	-32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
	-23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
	-15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
	-11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
	-7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
	-5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
	-3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
	-2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
	-1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
	-1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
	-876, -844, -812, -780, -748, -716, -684, -652,
	-620, -588, -556, -524, -492, -460, -428, -396,
	-372, -356, -340, -324, -308, -292, -276, -260,
	-244, -228, -212, -196, -180, -164, -148, -132,
	-120, -112, -104, -96, -88, -80, -72, -64,
	-56, -48, -40, -32, -24, -16, -8, 0,
	32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
	23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
	15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
	11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
	7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
	5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
	3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
	2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
	1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
	1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
	876, 844, 812, 780, 748, 716, 684, 652,
	620, 588, 556, 524, 492, 460, 428, 396,
	372, 356, 340, 324, 308, 292, 276, 260,
	244, 228, 212, 196, 180, 164, 148, 132,
	120, 112, 104, 96, 88, 80, 72, 64,
	56, 48, 40, 32, 24, 16, 8, 0,
}

// LinearToMulaw converts a 16-bit linear sample to 8-bit μ-law
func LinearToMulaw(sample int16) byte {
	sign := (sample >> 8) & 0x80
	if sign != 0 {
		sample = -sample
	}
	if sample > MulawClip {
		sample = MulawClip
	}
	sample = sample + MulawBias

	exponent := MulawEncodeTable[(sample>>7)&0xFF]
	mantissa := (sample >> (exponent + 3)) & 0x0F

	mulawByte := ^(int16(sign) | (exponent << 4) | mantissa)
	return byte(mulawByte)
}

// MulawToLinear converts an 8-bit μ-law sample to 16-bit linear
func MulawToLinear(mulawByte byte) int16 {
	return MulawDecodeTable[mulawByte]
}

// EncodeMulaw converts PCM16 audio to μ-law
func EncodeMulaw(pcm []int16) []byte {
	mulaw := make([]byte, len(pcm))
	for i, sample := range pcm {
		mulaw[i] = LinearToMulaw(sample)
	}
	return mulaw
}

// DecodeMulaw converts μ-law audio to PCM16
func DecodeMulaw(mulaw []byte) []int16 {
	pcm := make([]int16, len(mulaw))
	for i, sample := range mulaw {
		pcm[i] = MulawToLinear(sample)
	}
	return pcm
}

// EncodeMulawBytes converts PCM16 bytes (little-endian) to μ-law
func EncodeMulawBytes(pcm []byte) []byte {
	if len(pcm)%2 != 0 {
		pcm = pcm[:len(pcm)-1]
	}

	mulaw := make([]byte, len(pcm)/2)
	for i := 0; i < len(pcm); i += 2 {
		sample := int16(pcm[i]) | int16(pcm[i+1])<<8
		mulaw[i/2] = LinearToMulaw(sample)
	}
	return mulaw
}

// DecodeMulawBytes converts μ-law to PCM16 bytes (little-endian)
func DecodeMulawBytes(mulaw []byte) []byte {
	pcm := make([]byte, len(mulaw)*2)
	for i, sample := range mulaw {
		linear := MulawToLinear(sample)
		pcm[i*2] = byte(linear)
		pcm[i*2+1] = byte(linear >> 8)
	}
	return pcm
}

// EnsureMulaw checks if data is mulaw and converts if needed
// Heuristic: mulaw data has high byte values (centered around 0xFF for silence)
func EnsureMulaw(data []byte) []byte {
	if len(data) == 0 {
		return data
	}

	// If data looks like PCM (even length, varied byte distribution)
	// convert it to mulaw
	if len(data)%2 == 0 && !looksLikeMulaw(data) {
		return EncodeMulawBytes(data)
	}

	return data
}

// looksLikeMulaw checks if data appears to be mulaw encoded
func looksLikeMulaw(data []byte) bool {
	if len(data) < 10 {
		return true // Can't tell, assume mulaw
	}

	// Mulaw silence is 0xFF (255) or 0x7F (127)
	// PCM silence is 0x00
	highCount := 0
	for i := 0; i < min(100, len(data)); i++ {
		if data[i] > 0x70 {
			highCount++
		}
	}

	// If most values are high, likely mulaw
	return highCount > len(data)/3
}

// Resample converts audio from one sample rate to another using linear interpolation
func Resample(input []int16, fromRate, toRate int) []int16 {
	if fromRate == toRate {
		return input
	}

	ratio := float64(fromRate) / float64(toRate)
	outputLen := int(float64(len(input)) / ratio)
	output := make([]int16, outputLen)

	for i := 0; i < outputLen; i++ {
		srcPos := float64(i) * ratio
		srcIdx := int(srcPos)
		frac := srcPos - float64(srcIdx)

		if srcIdx+1 < len(input) {
			// Linear interpolation
			sample := float64(input[srcIdx])*(1-frac) + float64(input[srcIdx+1])*frac
			output[i] = int16(sample)
		} else if srcIdx < len(input) {
			output[i] = input[srcIdx]
		}
	}

	return output
}

// ResampleBytes resamples PCM16 bytes
func ResampleBytes(input []byte, fromRate, toRate int) []byte {
	if fromRate == toRate {
		return input
	}

	// Convert to int16
	samples := make([]int16, len(input)/2)
	for i := 0; i < len(samples); i++ {
		samples[i] = int16(input[i*2]) | int16(input[i*2+1])<<8
	}

	// Resample
	resampled := Resample(samples, fromRate, toRate)

	// Convert back to bytes
	output := make([]byte, len(resampled)*2)
	for i, sample := range resampled {
		output[i*2] = byte(sample)
		output[i*2+1] = byte(sample >> 8)
	}

	return output
}

// CalculateRMS calculates the root mean square of audio samples
func CalculateRMS(samples []int16) float64 {
	if len(samples) == 0 {
		return 0
	}

	var sum float64
	for _, sample := range samples {
		sum += float64(sample) * float64(sample)
	}

	return math.Sqrt(sum / float64(len(samples)))
}

// CalculateRMSBytes calculates RMS from PCM16 bytes
func CalculateRMSBytes(data []byte) float64 {
	samples := make([]int16, len(data)/2)
	for i := 0; i < len(samples); i++ {
		samples[i] = int16(data[i*2]) | int16(data[i*2+1])<<8
	}
	return CalculateRMS(samples)
}

// IsSilence checks if audio is silence based on RMS threshold
func IsSilence(samples []int16, threshold float64) bool {
	return CalculateRMS(samples) < threshold
}

// IsSilenceBytes checks if audio bytes are silence
func IsSilenceBytes(data []byte, threshold float64) bool {
	return CalculateRMSBytes(data) < threshold
}

// NormalizeSamples normalizes audio to a target level
func NormalizeSamples(samples []int16, targetLevel float64) []int16 {
	rms := CalculateRMS(samples)
	if rms == 0 {
		return samples
	}

	gain := targetLevel / rms
	output := make([]int16, len(samples))

	for i, sample := range samples {
		normalized := float64(sample) * gain
		// Clip to int16 range
		if normalized > 32767 {
			normalized = 32767
		} else if normalized < -32768 {
			normalized = -32768
		}
		output[i] = int16(normalized)
	}

	return output
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
