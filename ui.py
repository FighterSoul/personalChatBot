import tkinter as tk
from tkinter import Scrollbar, Text
from chatbot import predict_class

def send_message():
    user_message = entry.get()
    chat_history.insert(tk.END, "You: " + user_message + "\n")

    # Process the user message and get the bot's response
    ints = predict_class(user_message)
    bot_response = get_response(ints, intents)

    # Display the bot's response in the chat window
    chat_history.insert(tk.END, "Bot: " + bot_response + "\n")
    entry.delete(0, tk.END)  # Clear the input field

# Creating the main window
root = tk.Tk()
root.title("Chatbot GUI")

# Create a chat history window with a scrollbar
chat_history = Text(root, wrap="word", width=40, height=10)
scrollbar = Scrollbar(root, command=chat_history.yview)
chat_history.config(yscrollcommand=scrollbar.set)

# Create an entry widget for user input
entry = tk.Entry(root, width=30)

# Create a button to send the user's message
send_button = tk.Button(root, text="Send", command=send_message)

# Place the widgets on the window
chat_history.pack(padx=10, pady=10)
scrollbar.pack(side="right", fill="y")
entry.pack(pady=10)
send_button.pack()

# Run the Tkinter event loop
root.mainloop()
