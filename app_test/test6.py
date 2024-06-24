import tkinter as tk
import os
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence
import cv2

width = 800
height = 600
pad = 50
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir,"resource","image")
pig_image = os.path.join(image_path,"pig.png")
background_image = os.path.join(image_path,"walk.gif")

def print_test():
    print("ㅋㅋ 눌림")

def print_entry(event=None):
    value = entry.get() 
    if value.strip() == "":
        messagebox.showwarning("경고", "값을 입력하세요!") 
    else:
        switch_to_main() 

def switch_to_main():
    start_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

def switch_to_start():
    main_frame.pack_forget()
    start_frame.pack(fill="both", expand=True)

def switch_to_menu1():
    main_frame.pack_forget()
    menu1_frame.pack(fill="both", expand=True)

def create_image_button(input_image,x,y,command):
    try:
        button_image = tk.PhotoImage(file=input_image)
        button = tk.Button(main_frame, image=button_image, command=command)
        button.image = button_image
        button.place(x=x, y=y)
    except tk.TclError:
        print(f"Failed to load image at {pig_image}")

class AnimatedGIF:
    def __init__(self, canvas, filepath, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.sequence = []
        self.load_sequence(filepath)
        self.image_id = None
        self.current_frame = 0
        self.animate()

    def load_sequence(self, filepath):
        im = Image.open(filepath)
        for frame in ImageSequence.Iterator(im):
            frame = frame.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.sequence.append(ImageTk.PhotoImage(frame))

    def animate(self):
        if self.sequence:
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(self.x, self.y, anchor='nw', image=self.sequence[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.sequence)
            self.canvas.after(50, self.animate)

class VideoCapture:
    def __init__(self, canvas, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠) 사용
        self.image_id = None
        self.current_frame = None  # 현재 프레임을 저장할 변수
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()  # 현재 프레임을 변수에 저장
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            frame = cv2.resize(frame, (self.width, self.height))  # 프레임 크기 조정
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(self.x, self.y, anchor='nw', image=imgtk)
            self.canvas.image = imgtk
        self.canvas.after(10, self.update)  # 10ms마다 업데이트

    def get_current_frame(self):
        return self.current_frame

class VideoCapture:
    def __init__(self, canvas, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠) 사용
        self.image_id = None
        self.current_frame = None  # 현재 프레임을 저장할 변수
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()  # 현재 프레임을 변수에 저장
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            frame = cv2.resize(frame, (self.width, self.height))  # 프레임 크기 조정
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(self.x, self.y, anchor='nw', image=imgtk)
            self.canvas.image = imgtk
        self.canvas.after(10, self.update)  # 10ms마다 업데이트

    def get_current_frame(self):
        return self.current_frame

# tkinter 윈도우 생성
window = tk.Tk()
window.title("모드 전환 예제")
window.geometry(f"{width}x{height}")

# 시작 모드 프레임 생성
start_frame = tk.Frame(window)
start_frame.pack(fill="both", expand=True)

# 레이블 생성
label = tk.Label(start_frame, text="사용자명을 입력하세요:")
label.pack(pady=20)

# 입력 필드 생성
entry = tk.Entry(start_frame)
entry.pack(pady=10)

# Enter 키로도 입력 확인 가능하게 설정 
entry.bind("<Return>", print_entry)
button = tk.Button(start_frame, text="입력 확인", command=print_entry)
button.pack(pady=10)

# 메인 모드 프레임 생성
main_frame = tk.Frame(window)

# Canvas 생성 및 배경 이미지 설정
canvas = tk.Canvas(main_frame, width=width, height=height)
canvas.pack(fill="both", expand=True)
animated_bg = AnimatedGIF(canvas, background_image, 0, 0, width, height)

# 버튼 생성
button1 = create_image_button(pig_image, pad, pad, switch_to_start)
button2 = create_image_button(pig_image, height - pad, pad, switch_to_menu1)
button3 = create_image_button(pig_image, pad, height - 300, print_test)
button4 = create_image_button(pig_image, height - pad, height - 300, print_test)

menu1_frame = tk.Frame(window)
canvas = tk.Canvas(menu1_frame, width=width, height=height)
canvas.pack(fill="both", expand=True)
video_capture = VideoCapture(canvas, width // 4, height // 4, width // 2, height // 2)

# tkinter 윈도우 실행
window.mainloop()