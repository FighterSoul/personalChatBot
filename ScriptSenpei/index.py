import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    messages = [
        {"role": "system", "content": "You are a helpful and kind AI Assistant."},
    ]
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        return reply

inputs = gr.Textbox(lines=7, label="Chat with AI")
outputs = gr.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="default").launch(share=True)