import tkinter as tk
from tkinter import Canvas
import time

def create_encoder_window():
    encoder_window = tk.Toplevel()
    encoder_window.title("Encoder Details")

    canvas = Canvas(encoder_window, width=600, height=500)
    canvas.pack()

    # Draw the layers of the encoder
    for i in range(3):
        y_start = 50 + i * 150
        y_end = y_start + 130
        canvas.create_rectangle(50, y_start, 550, y_end, fill="lightblue", outline="black")

        # Multi-head attention
        canvas.create_rectangle(70, y_start + 10, 260, y_start + 60, fill="yellow", outline="black")
        canvas.create_text(165, y_start + 35, text="Multi-Head Attention")

        # Residual connection
        canvas.create_line(260, y_start + 35, 340, y_start + 35, arrow=tk.LAST)

        # Feed-forward network
        canvas.create_rectangle(340, y_start + 10, 530, y_start + 60, fill="orange", outline="black")
        canvas.create_text(435, y_start + 35, text="Feed-Forward Network")

        # Layer normalization
        canvas.create_text(60, y_start + 80, text="Layer Norm", anchor="w")

        # Residual connection from layer normalization
        if i < 2:
            canvas.create_line(290, y_start + 110, 290, y_start + 150, arrow=tk.LAST)

    # Labels
    canvas.create_text(300, 20, text="Encoder Layers", font=("Arial", 16))

def main():
    root = tk.Tk()
    root.title("Transformer Visualization")

    main_canvas = Canvas(root, width=800, height=100)
    main_canvas.pack()
    main_canvas.create_text(400, 50, text="Main Transformer Visualization Window")

    # Button to open the encoder details window
    encoder_button = tk.Button(root, text="Show Encoder Details", command=create_encoder_window)
    encoder_button.pack()

    root.mainloop()

main()
