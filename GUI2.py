import tkinter as tk
root=tk.Tk()
root.title("how to do ")
root.geometry('500x300')
tk.Label(root, text="显示", bg="green", font=("Arial", 12), width=5, height=1).place(x=20,y=30)

def printtext():
    EditText.insert(1.0,A())
def A():
    if 2>3:
        return "句子1"
    elif 2<0:
        return "句子2"
    else:
        return "句子3"

EditText = tk.Text(root,width=20,height=10)
EditText.grid(row=2,column=3)
btn_test=tk.Button(root, text="按钮", command =printtext,width=5, height=2)

btn_test.place( x=200,y=60)

root.mainloop()