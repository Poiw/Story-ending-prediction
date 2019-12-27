from configparser import ConfigParser
import argparse

def parse_arg():

    parser = argparse.ArgumentParser(description='trainscript.py') # ???

    # args to set - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--maxSentenceLength', type=int, default=20)
    parser.add_argument('--embeddingmodel', type=str, default='../Models/embedding.model')


    parser.add_argument('--BS', type=int, default=64)

    # return the params
    opt = parser.parse_args()
    return opt

options = parse_arg()
print('options', options)
print('done.')
