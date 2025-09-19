#
#import tkinter as tk
#from tkinter import scrolledtext




# Search and answer using GPT
def search_and_respond():
    query = input_box.get("1.0", tk.END).strip()
    if not query:
        return

    try:
        embedding = embed_user_query(query)
        results = search_faiss(embedding, top_k=5)
        context = "\n\n".join([r["text"] for r in results])
        hybrid_query = f"{context}\n\n---\n\nQuestion: {query}"

        # Primer placed inside as you requested
        primer = """You are an expert assistant specialized in airline revenue management and fare rule violations.
    Answer questions only based on the provided context (e.g., IATA fare rules, category restrictions, GDS misuse, ADM risk).
    If the answer is not in the context, say: "I don't know, sorry." Never invent information."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": hybrid_query}
            ]
        )

        answer = response.choices[0].message.content


    except Exception as e:
        answer = f"❌ Error: {e}"

    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, answer)
    output_box.config(state="disabled")

# 🖼️ Build Tkinter GUI
def launch_chatbot():
    root = tk.Tk()
    root.title("Airline Fare Violation Assistant")

    tk.Label(root, text="Enter your question:").pack()
    global input_box
    input_box = tk.Text(root, height=4, width=80)
    input_box.pack()

    tk.Button(root, text="Search", command=search_and_respond).pack(pady=5)

    tk.Label(root, text="Answer:").pack()
    global output_box
    output_box = scrolledtext.ScrolledText(root, height=15, width=80, state="disabled", wrap=tk.WORD)
    output_box.pack()

    root.mainloop()

    ###########################################################################################################################
    chat history

    '''        if not is_strict_factual:
                # Only include last 4 messages (2 user+assistant pairs)
                valid_turns = []
                for msg in reversed(chat_history):
                    # Only user or assistant with meaningful answer
                    if msg["role"] == "user":
                        valid_turns.insert(0, msg)
                    elif msg["role"] == "assistant" and "i don't know" not in msg["content"].lower():
                        valid_turns.insert(0, msg)
                    if len(valid_turns) >= 4:
                        break
                for msg in valid_turns:
                    role_name = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role_name}: {msg['content']}\n"
    '''