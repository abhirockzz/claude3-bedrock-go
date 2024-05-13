package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
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
const modelID = "anthropic.claude-3-sonnet-20240229-v1:0"

func main() {
	verbose = flag.Bool("verbose", false, "setting to true will log messages being exchanged with LLM")
	flag.Parse()

	reader := bufio.NewReader(os.Stdin)

	payload := Claude3Request{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        1024,
	}

	for {
		fmt.Print("\nEnter your message: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		msg := Message{
			Role: userRole,
			Content: []Content{
				{
					Type: contentTypeText,
					Text: input,
				},
			},
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
	Type string `json:"type,omitempty"`
	Text string `json:"text,omitempty"`
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
