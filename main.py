from utils.preprocessing import *
from utils.pseudo_abscence import *
from utils.heatmap import *
from utils.random_utils import *
from utils.models import *



if __name__ == '__main__':
    while True:
        ##Main process

        ##output program
        while True:
            ans = input('Do you want to create another distribution or quit:(q: quit, c:continue)').lower()
            if ans is not 'c' or 'q':
                continue
            else: 
                break