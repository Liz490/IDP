"""Script to convert csv to json files. """

import json

paths = [
    "/nfs/home/students/l.schmierer/code/IDP/logs/hparams/Version_1/18.09-08:43",
    "/nfs/home/students/l.schmierer/code/IDP/logs/hparams/Version_1/18.09-19:59",
    "/nfs/home/students/l.schmierer/code/IDP/logs/hparams/Version_1/19.09-02:15",
    "/nfs/home/students/l.schmierer/code/IDP/logs/hparams/Version_1/19.09-04:01",
    "/nfs/home/students/l.schmierer/code/IDP/logs/hparams/Version_1/19.09-04:21"
]

for infilename in paths:
    d = {}
    for line in open(infilename):
        k, *v = line.split(',')
        v = ','.join(v).replace("'", '"').replace("False", "false").replace("True", "true")
        try:
            if v.startswith('"'): v = v[1:-2]
            v = json.loads(v)
        except json.decoder.JSONDecodeError:
            v = v.strip()
        d[k] = v
    outfilename = infilename + '.json'
    json.dump(d, open(outfilename, 'w'))
