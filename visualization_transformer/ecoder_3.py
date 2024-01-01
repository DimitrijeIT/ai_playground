import tkinter as tk
from tkinter import Canvas
import time

def animate_attention(canvas):
    # Animation for query, key, value, and attention calculation
    for i in range(3):
        # Simulate data for Query, Key, Value
        qkv_y = 50 + i * 100
        for label, x in zip(["Query", "Key", "Value"], [50, 250, 450]):
            canvas.create_rectangle(x, qkv_y, x + 150, qkv_y + 50, fill="lightblue", outline="black")
            canvas.create_text(x + 75, qkv_y + 25, text=label)
            data = canvas.create_oval(x + 65, qkv_y + 15, x + 85, qkv_y + 35, fill="red", tags="data")
            canvas.update()
            time.sleep(0.5)

            # Move to the attention calculation
            for _ in range(10):
                canvas.move(data, 10, 5)
                canvas.update()
                time.sleep(0.1)

            canvas.delete(data)

        # Simulate attention calculation
        canvas.create_rectangle(600, qkv_y, 750, qkv_y + 50, fill="yellow", outline="black")
        canvas.create_text(675, qkv_y + 25, text="Attention")
        canvas.update()
        time.sleep(1)

def create_attention_window():
    attention_window = tk.Toplevel()
    attention_window.title("Multi-Head Attention Details")

    canvas = Canvas(attention_window, width=800, height=300)
    canvas.pack()

    # Label
    canvas.create_text(400, 20, text="Multi-Head Attention Process", font=("Arial", 16))

    # Start animation
    animate_attention(canvas)

def main():
    root = tk.Tk()
    root.title("Transformer Visualization")

    main_canvas = Canvas(root, width=800, height=100)
    main_canvas.pack()
    main_canvas.create_text(400, 50, text="Main Transformer Visualization Window")

    # Button to open the multi-head attention details window
    attention_button = tk.Button(root, text="Show Multi-Head Attention Details", command=create_attention_window)
    attention_button.pack()

    root.mainloop()

main()
