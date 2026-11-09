"""Compare serving/eval commands while allowing only per-run addresses and outputs."""
import json


def normalize(command):
    command=list(command)
    for flag in ('--port','--base-url','--out'):
        if flag in command:
            command[command.index(flag)+1]='<run-specific>'
    if '--extra-body' in command:
        i=command.index('--extra-body')+1
        body=json.loads(command[i]);body.pop('cache_salt',None)
        command[i]=json.dumps(body,sort_keys=True,separators=(',',':'))
    return command


def assert_equivalent(current,reference):
    if normalize(current)!=normalize(reference):
        raise ValueError('Serving/eval command differs from the completed E074 reference beyond address/output/cache_salt')
