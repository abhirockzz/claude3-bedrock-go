package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
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

// const modelID = "anthropic.claude-3-sonnet-20240229-v1:0"
const modelID = "anthropic.claude-3-haiku-20240307-v1:0"

func main() {

	// msg := "If I buy all the items in the menu, how much would it cost me?"
	// imagePath := "menu.jpg"

	msg := "Transcribe the code in the question. Only output the code."
	imagePath := "soflow.jpg"

	// msg := "Can you suggest a solution to the question?"
	// imagePath := "soflow.jpg"

	imageContents, err := readImageAsBase64(imagePath)
	if err != nil {
		log.Fatal(err)
	}

	payload := Claude3Request{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        1024,
		Messages: []Message{
			{
				Role: "user",
				Content: []Content{
					{
						Type: "image",
						Source: &Source{
							Type:      "base64",
							MediaType: "image/jpeg",
							Data:      imageContents,
						},
					},
					{
						Type:   "text",
						Text:   msg,
						Source: nil,
					},
				},
			},
		},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}
	//fmt.Println("request payload:\n", string(payloadBytes))

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		log.Fatal(err)
	}

	var resp Claude3Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal(err)
	}

	//fmt.Println("response payload:\n", string(output.Body))

	fmt.Println("response string:\n", resp.ResponseContent[0].Text)

}

func readImageAsBase64(filePath string) (string, error) {
	imageFile, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	encodedString := base64.StdEncoding.EncodeToString(imageFile)

	return encodedString, nil
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
type Source struct {
	Type      string `json:"type,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}
type Content struct {
	Type   string  `json:"type,omitempty"`
	Source *Source `json:"source,omitempty"`
	Text   string  `json:"text,omitempty"`
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
