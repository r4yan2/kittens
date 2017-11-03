#!/bin/bash

disclaimer="
            --> Kitt<3ns Recommendation ENGINE <--

      Refer to the README for documentation and example usage

      Please select your choice wisely...
"

running="
         Running...
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

# if pypy is not installed ask the user the permission to install it else terminate
command -v pypy >/dev/null
if [[ "$?" == "1" ]]; then
  whiptail --yesno "pypy is required, proceed with installation?" 10 40
  if [[ "$?" == "0" ]]; then
    sudo apt install pypy
  else
    exit 0
  fi
fi

# calculate the amount of free memory
mem=$(free | awk 'FNR == 2 {print int($7/(1024*1024*1.5))}')

if [[ "$mem" == "0" ]]; then
  whiptail --msgbox "Sorry, the system does not have enough memory available (1.5Gb minimum is required)" 10 50
  exit 0
fi

# get the total number of process of this machine
cpu=$(nproc --all)

# select the amount of core to use core = min(available_memory mod 2GB, available_cpu)
core=$(($mem<$cpu?$mem:$cpu))

whiptail --msgbox "$disclaimer" 15 70

# ask the user to select operative mode
mode=$(whiptail --menu "Select Operational Mode" 15 70 5 \
0 "Test Mode" \
1 "Recommendation Mode" \
2 "Script Mode" \
3 "Debug Mode" \
3>&2 2>&1 1>&3)

# if user has selected "Cancel" then exit
if [[ -z "$mode" ]]; then exit 0; fi

case $mode in
    2)
        script=$(whiptail --menu "Which script do you want to run?" 20 80 5 \
        0 "Compute test set (x3)" \
        1 "Compute neighborhood" \
        3>&2 2>&1 1>&3)
        
        # if user has selected "Cancel" then exit
        if [[ -z "$script" ]]; then exit 0; fi
        case $script in
          0)
            /usr/bin/pypy test_set_generator.py
          ;;
          1)
            /usr/bin/pypy neighborhood_generator.py
          ;;
        esac
    ;;
    *)
        # ask the user to select a recommendations method
        recommendations=$(whiptail --menu "Select Recommendations Method" 25 70 18 \
        0 "Random" \
        1 "Top Listened" \
        2 "Top Included" \
        3 "Top Tags" \
        4 "TF-IDF based on tags" \
        5 "Top-Tag combined TfIdf" \
        6 "TfIdf combined Top-Tag" \
        7 "TfIdf based on titles" \
        8 "Tfidf tags combined tfIdf titles" \
        9 "Top-Tag combined TfIdf titles" \
        10 "Tfidf Titles filter applied on normal tfidf" \
        11 "Collaborative item-item recommendations" \
        12 "Artist Recommendations + tfidf padding" \
        13 "Hybrid recommendations" \
        14 "Neighborhood similarity recommendations" \
        15 "User based recommendations" \
        16 "Naive Bayes computation" \
        3>&2 2>&1 1>&3)

        # if user has selected "Cancel" then exit
        if [[ -z "$recommendations" ]]; then exit 0; fi

        # if in debug mode no progress bar is showed, so print could be shown
        if [[ "$mode" == "3" ]]
        then /usr/bin/pypy kittens.py 0 "$recommendations" 1 1
        else /usr/bin/pypy kittens.py "$mode" "$recommendations" "$core" 1 | whiptail --gauge "$running" 15 60 0
        fi
    ;;
esac
case $mode in
    0)
        # end of test mode with test results on screen
        whiptail --textbox data/test_result1.csv 12 60
    ;;
    3)
        # end of debug mode
        exit 0
    ;;
    *)
        # end with fancy message
        whiptail --msgbox "$end" 15 50
    ;;
esac
exit 0
