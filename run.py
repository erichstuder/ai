#!/usr/bin/env python3

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
        commands = 'jupyter nbconvert --to notebook --execute 1-1.ipynb --output-dir /tmp &&' \
                   'jupyter nbconvert --to notebook --execute 1-2-1.ipynb --output-dir /tmp'
    else:
        commands = None

    ex.run(commands)
