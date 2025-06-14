import os

simulation_template = '''#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/{jobname}.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/{jobname}.err

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield  # activate here the anaconda environment you want to use

python {script_path} {script_kwargs}
'''


DEFAULT_SBATCH_PARAMS = {
    'partition': 'hij-gpu',
    'n_cpus': 24,
    'time': '24:00:00',
    'mem': '100GB',
    'script_path': '/home/yu79deg/darkfield_p5438/src/darkfield/dfdf_Aime.py'
}


def write_bash(path, N, upd_params={}, bash_name=None):
    
    bash_params = DEFAULT_SBATCH_PARAMS.copy()

    if upd_params:
        bash_params.update(upd_params)
            
    #bash_params['script_kwargs'] = f'-N {N}'
    
    if 'yaml' not in upd_params:
        raise ValueError("You must specify 'yaml' in upd_params.")
        
    bash_params['script_kwargs'] = f'-N {N} --yaml {upd_params["yaml"]}'

    yaml_file = upd_params['yaml']
    bash_params['jobname'] = os.path.splitext(os.path.basename(yaml_file))[0] #extracting the job name from the name of the yaml file.

    bash_script = simulation_template.format(**bash_params)

    if bash_name is None:
        bash_path = os.path.join(path, 'job.slurm')
    else:
        bash_path = os.path.join(path, bash_name)
        
    with open(bash_path, 'w') as f:
        f.write(bash_script)
    return bash_path