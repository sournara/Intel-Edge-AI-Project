import tkinter as tk
import os
from tkinter import messagebox


width = 800
height = 600
pad = 50
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir,"resource","image")
pig_image = os.path.join(image_path,"pig.png")

# 버튼 클릭 시 새로운 창을 열기 위한 함수
def open_new_window():
    # 메인 윈도우 비활성화
    window.withdraw()

    new_window = tk.Toplevel()
    new_window.title("새 창")
    label = tk.Label(new_window, text="새로운 창이 열렸습니다!")
    label.pack(padx=20, pady=20)

    def close_new_window():
        new_window.destroy()
        # 메인 윈도우 다시 보이기
        window.deiconify()
    # 새 창 닫기 버튼 추가
    close_button = tk.Button(new_window, text="닫기", command=close_new_window)
    close_button.pack(pady=10)

def open_new_menu(title):
    window.withdraw()
    new_window = tk.Toplevel()
    new_window.title(f"{title}")
    
    def close_new_window():
        new_window.destroy()
        # 메인 윈도우 다시 보이기
        window.deiconify()

    close_button = tk.Button(new_window, text="닫기", command=close_new_window)
    close_button.pack(pady=10)

def print_test():
    print("ㅋㅋ 눌림")

def create_image_button(input_image,x,y,command):
    try:
        button_image = tk.PhotoImage(file=input_image)
        button = tk.Button(window, image=button_image, command=command)
        button.image = button_image
        button.place(x=x, y=y)
    except tk.TclError:
        print(f"Failed to load image at {pig_image}")


# tkinter 윈도우 생성
window = tk.Tk()
window.title("하이 아임 지훈")
window.geometry(f"{width}x{height}")
# button = tk.Button(window, text="새 창 열기", command=open_new_window)
# button.pack(pady=20)


# 버튼 생성
button1 = create_image_button(pig_image, pad , pad,open_new_window)
button2 = create_image_button(pig_image, height - pad , pad,print_test)
button3 = create_image_button(pig_image, pad , height - 300,open_new_window)
button4 = create_image_button(pig_image, height - pad , height - 300,open_new_window)


# GUI 루프 시작
window.mainloop()


