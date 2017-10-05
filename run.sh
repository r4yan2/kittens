#!/bin/bash

disclaimer="
            --> Kitt<3ns Recommendation ENGINE <--

    Capabilities:

    Random Recommendations
    Top Listened Recommendations
    Top Included Recommendations
    Top Tags Recommendations

    Test Mode

    Please wait until the Engine is ready, then select your choice
"

running="
         VROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM
         Parallel Engine ACTIVATION
         CORE TRIGGERED
         CORE TRIGGERED
         CORE TRIGGERED
         CORE TRIGGERED
         VROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM
"

end="
         Completed!
         Result Writed to file correctly
         Bye
"

end_test="
         Test Mode Completed!
         Average Accuracy $value
"

whiptail --msgbox "$disclaimer" 20 70
whiptail --yesno "Enable Test Mode?" 10 50
test=$?
choice=$(whiptail --menu "Select Recommendations Method" 20 70 5 \
0 "Random" \
1 "Top Listened" \
2 "Top Included" \
3 "Top Tags" 3>&2 2>&1 1>&3)
case $choice in
    0)
        /usr/bin/python kittens.py "$test" 0 | whiptail --gauge "$running" 10 60 0
    ;;
    *)
        /usr/bin/pypy kittens.py "$test" "$choice" | whiptail --gauge "$running" 10 60 0
    ;;
esac
case $test in
    1)
        whiptail --msgbox "$end" 20 50
    ;;
    0)
        value=`cat data/result.csv`
        whiptail --msgbox "$end_test" 20 50
    ;;
esac
exit 0

