#!/usr/bin/env python3

import os
from project_management import Executor

if __name__ == "__main__":
    additional_arguments = [
        {
            'flag': '-t',
            'name': '--test',
            'help': 'Run unit tests.'
        },
        {
            'flag': '-n',
            'name': '--test_notebook',
            'help': 'Run notebook to see if it works.'
        }
    ]

    ex = Executor(additional_arguments, description='Run tests.')

    if ex.arguments.test:
        commands = 'pytest'
    elif ex.arguments.test_notebook:
        commands = ''
        for filename in os.listdir("notebooks"):
            commands += f'jupyter nbconvert --to notebook --execute notebooks/{filename} --output-dir /tmp && '
        commands = commands[:-4] # remove the last ' && '
    else:
        commands = None

    ex.run(commands)
