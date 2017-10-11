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
         Result writed to file correctly
         Bye
"

end_test="
         Test Mode Completed!
         Average Accuracy
"

whiptail --msgbox "$disclaimer" 20 70
mode=$(whiptail --menu "Select Operational Mode" 20 70 5 \
0 "Test Mode" \
1 "Recommendation Mode" \
2 "Script Mode" 3>&2 2>&1 1>&3)
case $mode in
    2)
        script=$(whiptail --menu "Which script do you want to run?" 20 70 5 \
        0 "Compute test set (x3)" 3>&2 2>&1 1>&3)
        /usr/bin/pypy script.py
    ;;
    *)
        recommendations=$(whiptail --menu "Select Recommendations Method" 20 70 5 \
        0 "Random" \
        1 "Top Listened" \
        2 "Top Included" \
        3 "Top Tags" \
        4 "TF-IDF based" 3>&2 2>&1 1>&3)
        /usr/bin/pypy kittens.py "$mode" "$recommendations" | whiptail --gauge "$running" 15 60 0
    ;;
esac
case $mode in
    0)
        test_results=$(cat data/test_result* | awk '{if(min==""){min=max=$1}; if($1>max) {max=$1}; if($1<min) {min=$1}; total+=$1; count+=1} END {print "avg",total/count,"max",max,"min",min}')
        whiptail --msgbox "Test Mode Completed! ""$test_results" 15 50
    ;;
    *)
        whiptail --msgbox "$end" 20 50
    ;;
esac
exit 0
