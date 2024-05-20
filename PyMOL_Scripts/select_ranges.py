import pymol
import sys


def Select_Range(resn, letter, lower, upper, end=''):
    
    lower = int(lower)
    upper = int(upper)
    selection_name = f"{resn}_{letter}_{lower}_{upper}{end} "
    selection_expression = f"resn {resn} and name "
    for i in range(lower, upper+1):
        selection_expression += f'{letter}{i}{end}+'

    selection_expression = selection_expression[:-1]
    print(selection_expression)


    pymol.cmd.select(selection_name, selection_expression)

cmd.extend("SelRange", Select_Range)
