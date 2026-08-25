str = "Hello@@@@@World"
upperCount = 0
lowerCount = 0
for i in range(0, len(str)):
    if str[i] >= "a" and str[i] <= "z":
        # print(count)
        lowerCount = lowerCount + 1
    elif str[i] >= "A" and str[i] <= "Z":
        upperCount = upperCount + 1
    else:
        print("Invalid inpput")
print(lowerCount)
print(upperCount)
