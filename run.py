#!/usr/bin/env python3

from project_management import Executor

if __name__ == "__main__":
    additional_arguments = [
        {
            'flag': '-t',
            'name': '--test',
            'help': 'Run tests.'
        }
    ]

    ex = Executor(additional_arguments, description='Run tests.')

    if ex.arguments.test:
        commands = 'pytest'
    else:
        commands = None

    ex.run(commands)
