"""Compare IC loss integrals from the generated diagnostic Fortran variants."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'results/high_temperature_diagnosis/old_code'
import argparse
p=argparse.ArgumentParser()
p.add_argument('--compiler', default='gfortran')
args=p.parse_args()
compiler=args.compiler
flags=['-O2','-std=legacy','-ffixed-line-length-none']
if Path(compiler).is_absolute():
    lib=Path(compiler).parent.parent/'lib/gcc/x86_64-linux-gnu/15'
    flags += ['-B'+str(lib)+'/', '-B/usr/lib/gcc/x86_64-linux-gnu/15/', '-L'+str(lib)]
driver='''      program probe
      implicit none
      integer i,j
      real*8 temp,energy,rate
      real*8 ts(3),es(5)
      data ts/1.d-7,1.d-5,1.d-3/
      data es/3.d-3,1.d-2,1.d-1,1.d0,1.d2/
      do i=1,3
        do j=1,5
          temp=ts(i)
          energy=es(j)
          call ic_v_ele(temp,energy,rate)
          write(*,'(3ES25.16)') temp,energy,rate
        enddo
      enddo
      end
'''
for mode in ['original','dp_current_ic_loss']:
    s=(root/f'{mode}.f').read_text();s=s[s.index('      subroutine spectrum('):]
    p=root/f'probe_{mode}.f';p.write_text(driver+s)
    exe=root/f'probe_{mode}'
    subprocess.run([compiler,*flags,str(p),'-o',str(exe)],check=True)
    with (root/f'ic_loss_{mode}.dat').open('w') as f: subprocess.run([str(exe)],stdout=f,check=True)
