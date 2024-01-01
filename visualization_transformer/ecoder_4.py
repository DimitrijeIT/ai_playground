import tkinter as tk
from tkinter import Canvas
import time

def animate_attention_heads(canvas):
    # Function to animate the attention heads
    def animate_head(x_start, y_start, color):
        # Animate Query, Key, Value
        for label, x_offset in zip(["Query", "Key", "Value"], [0, 60, 120]):
            data = canvas.create_oval(x_start + x_offset, y_start, x_start + x_offset + 20, y_start + 20, fill=color)
            canvas.update()
            time.sleep(0.5)
            for _ in range(10):
                canvas.move(data, 5, 5)
                canvas.update()
                time.sleep(0.1)
            canvas.delete(data)

        # Animate Attention calculation
        attn = canvas.create_oval(x_start + 60, y_start + 100, x_start + 80, y_start + 120, fill=color)
        canvas.update()
        time.sleep(1)
        canvas.delete(attn)

    # Draw the layout for multi-head attention
    head_colors = ["red", "green", "blue", "purple"]
    for i, color in enumerate(head_colors):
        x_start = 50 + i * 180
        canvas.create_text(x_start + 60, 30, text=f"Head {i+1}", font=("Arial", 12))
        animate_head(x_start, 50, color)

    # Combine outputs of heads
    combined_output = canvas.create_rectangle(50, 300, 750, 350, fill="grey", outline="black")
    canvas.create_text(400, 325, text="Combined Output of Heads")
    canvas.update()
    time.sleep(1)

def create_attention_window():
    attention_window = tk.Toplevel()
    attention_window.title("Multi-Head Attention Details")

    canvas = Canvas(attention_window, width=800, height=400)
    canvas.pack()

    # Label
    canvas.create_text(400, 10, text="Multi-Head Attention Mechanism", font=("Arial", 16))

    # Start animation
    animate_attention_heads(canvas)

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
