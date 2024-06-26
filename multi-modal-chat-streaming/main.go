package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const defaultRegion = "us-east-1"

var brc *bedrockruntime.Client

func init() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc = bedrockruntime.NewFromConfig(cfg)
}

var verbose *bool

const userRole = "user"
const assistantRole = "assistant"
const contentTypeText = "text"

// const modelID = "anthropic.claude-3-sonnet-20240229-v1:0"
const modelID = "anthropic.claude-3-haiku-20240307-v1:0"

func main() {
	verbose = flag.Bool("verbose", false, "setting to true will log messages being exchanged with LLM")
	flag.Parse()

	reader := bufio.NewReader(os.Stdin)

	payload := Claude3Request{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        1024,
	}

	for {
		fmt.Print("\nChoose your message type - Text (enter 1) or Image (enter 2): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		msg := Message{
			Role: userRole,
		}

		if input == "1" {

			fmt.Print("\nEnter your message: ")
			text, _ := reader.ReadString('\n')
			text = strings.TrimSpace(text)

			textContent := Content{
				Type: "text",
				Text: text,
			}
			msg.Content = append(msg.Content, textContent)

		} else if input == "2" {

			for {
				fmt.Print("\nEnter the image source (local path or url): ")
				path, _ := reader.ReadString('\n')
				path = strings.TrimSpace(path)

				imageContents, err := readImageAsBase64(path)
				if err != nil {
					log.Fatal(err)
				}

				imageContent := Content{Type: "image", Source: &Source{
					Type:      "base64",
					MediaType: "image/jpeg",
					Data:      imageContents,
				}}
				msg.Content = append(msg.Content, imageContent)

				fmt.Print("\nWould you like to add more images? enter yes or no: ")
				yesOrNo, _ := reader.ReadString('\n')
				yesOrNo = strings.TrimSpace(yesOrNo)

				if yesOrNo == "no" {
					fmt.Print("\nWhat would you like to ask about the image(s)? : ")
					q, _ := reader.ReadString('\n')
					q = strings.TrimSpace(q)

					textContent := Content{
						Type: "text",
						Text: q,
					}
					msg.Content = append(msg.Content, textContent)

					break
				} else if yesOrNo == "yes" {
					continue
				} else {
					log.Fatal("invalid option. enter yes or no. start over again")
				}
			}

		} else {
			log.Fatal("invalid option. enter 1 or 2. start over again")
		}

		payload.Messages = append(payload.Messages, msg)

		response, err := send(payload)

		if err != nil {
			log.Fatal(err)
		}

		//fmt.Println("[Assistant]:", response)

		respMsg := Message{
			Role: assistantRole,
			Content: []Content{
				{
					Type: contentTypeText,
					Text: response,
				},
			},
		}
		payload.Messages = append(payload.Messages, respMsg)

	}
}

func send(payload Claude3Request) (string, error) {

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	if *verbose {
		fmt.Println("[request payload]", string(payloadBytes))
	}

	output, err := brc.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        payloadBytes,
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return "", err
	}

	fmt.Print("[Assistant]: ")

	resp, err := processStreamingOutput(output, func(ctx context.Context, part []byte) error {
		fmt.Print(string(part))
		return nil
	})

	if err != nil {
		log.Fatal("streaming output processing error: ", err)
	}

	return resp.ResponseContent[0].Text, nil
}

type Claude3Request struct {
	AnthropicVersion string    `json:"anthropic_version"`
	MaxTokens        int       `json:"max_tokens"`
	Messages         []Message `json:"messages"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	TopK             int       `json:"top_k,omitempty"`
	StopSequences    []string  `json:"stop_sequences,omitempty"`
	SystemPrompt     string    `json:"system,omitempty"`
}

type Content struct {
	Type   string  `json:"type,omitempty"`
	Source *Source `json:"source,omitempty"`
	Text   string  `json:"text,omitempty"`
}
type Source struct {
	Type      string `json:"type,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}
type Message struct {
	Role    string    `json:"role,omitempty"`
	Content []Content `json:"content,omitempty"`
}

type Claude3Response struct {
	ID              string            `json:"id,omitempty"`
	Model           string            `json:"model,omitempty"`
	Type            string            `json:"type,omitempty"`
	Role            string            `json:"role,omitempty"`
	ResponseContent []ResponseContent `json:"content,omitempty"`
	StopReason      string            `json:"stop_reason,omitempty"`
	StopSequence    string            `json:"stop_sequence,omitempty"`
	Usage           Usage             `json:"usage,omitempty"`
}
type ResponseContent struct {
	Type string `json:"type,omitempty"`
	Text string `json:"text,omitempty"`
}
type Usage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

type PartialResponse struct {
	Type    string                 `json:"type"`
	Message PartialResponseMessage `json:"message,omitempty"`
	Index   int                    `json:"index,omitempty"`
	Delta   Delta                  `json:"delta,omitempty"`
	Usage   PartialResponseUsage   `json:"usage,omitempty"`
}

type PartialResponseMessage struct {
	ID           string               `json:"id,omitempty"`
	Type         string               `json:"type,omitempty"`
	Role         string               `json:"role,omitempty"`
	Content      []interface{}        `json:"content,omitempty"`
	Model        string               `json:"model,omitempty"`
	StopReason   string               `json:"stop_reason,omitempty"`
	StopSequence interface{}          `json:"stop_sequence,omitempty"`
	Usage        PartialResponseUsage `json:"usage,omitempty"`
}

type PartialResponseUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

type Delta struct {
	Type       string `json:"type,omitempty"`
	Text       string `json:"text,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
}

const partialResponseTypeContentBlockDelta = "content_block_delta"
const partialResponseTypeMessageStart = "message_start"
const partialResponseTypeMessageDelta = "message_delta"

type StreamingOutputHandler func(ctx context.Context, part []byte) error

func processStreamingOutput(output *bedrockruntime.InvokeModelWithResponseStreamOutput, handler StreamingOutputHandler) (Claude3Response, error) {

	var combinedResult string
	resp := Claude3Response{
		Type:            "message",
		Role:            "assistant",
		Model:           "claude-3-sonnet-28k-20240229",
		ResponseContent: []ResponseContent{{Type: contentTypeText}}}

	for event := range output.GetStream().Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:

			var pr PartialResponse
			err := json.NewDecoder(bytes.NewReader(v.Value.Bytes)).Decode(&pr)
			if err != nil {
				return resp, err
			}

			if pr.Type == partialResponseTypeContentBlockDelta {
				handler(context.Background(), []byte(pr.Delta.Text))
				combinedResult += pr.Delta.Text
			} else if pr.Type == partialResponseTypeMessageStart {
				resp.ID = pr.Message.ID
				resp.Usage.InputTokens = pr.Message.Usage.InputTokens
			} else if pr.Type == partialResponseTypeMessageDelta {
				resp.StopReason = pr.Delta.StopReason
				resp.Usage.OutputTokens = pr.Message.Usage.OutputTokens
			}

		case *types.UnknownUnionMember:
			fmt.Println("unknown tag:", v.Tag)

		default:
			fmt.Println("union is nil or unknown type")
		}
	}

	//resp.ResponseContent = []ResponseContent{}
	resp.ResponseContent[0].Text = combinedResult

	return resp, nil
}

func readImageAsBase64(source string) (string, error) {

	var imageBytes []byte

	if strings.Contains(source, "http") {
		resp, err := http.Get(source)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()

		imageBytes, err = io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}
	} else {
		//assume it's local
		var err error
		imageBytes, err = os.ReadFile(source)
		if err != nil {
			return "", err
		}
	}

	encodedString := base64.StdEncoding.EncodeToString(imageBytes)

	return encodedString, nil
}
