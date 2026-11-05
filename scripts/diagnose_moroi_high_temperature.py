"""Build isolated old-code variants; never edit moroi_old.f or production physics.

Compiler path may be supplied with --compiler. One temperature per process.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import json
import hashlib
import time

ROOT=Path(__file__).resolve().parents[1]


def run(task):
    exe, temperature, output = task
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        subprocess.run([exe,str(temperature)],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    return str(out),time.monotonic()-start


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compiler',default='gfortran')
    p.add_argument('--output',type=Path,default=ROOT/'results/high_temperature_diagnosis/old_code')
    p.add_argument('--temperatures',type=float,nargs='+',default=[1e-7,1e-5,1e-3])
    p.add_argument('--variants',nargs='+',default=['original','dp_endpoint_fixed'])
    args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    old=(ROOT/'moroi_old.f').read_text(encoding='latin1')
    prefix=old[:old.index('      ifpho=imesh-10')]
    prefix+='      character*80 arg\n'
    prefix+='''      call get_command_argument(1,arg)
      read(arg,*) temp
      eta=6.10E-10
      epho0=1.0d2
      ndotin=1.0d0
      ifpho=imesh-10
'''
    calc=old[old.index('          do jmesh=1,imesh\n'):old.index('c     --- process : 6li (g,x) x;')]
    driver=prefix+calc+'''      open(20,file='spectrum.dat',status='replace')
      do jmesh=1,imesh-1
        write(20,'(3ES25.16)') egy(jmesh),fpho(jmesh),fele(jmesh)
      enddo
      close(20)
      open(21,file='rates.dat',status='replace')
      write(21,'(11ES25.16)') eta,epho0,temp,ratea,rateb,
     & ratec,rated,ratee,ratef,rateg,ratei
      close(21)
      end
'''
    routines=old[old.index('      subroutine spectrum('):]
    compiler=Path(args.compiler)
    flags=['-O2','-std=legacy','-ffixed-line-length-none']
    # A locally extracted compiler uses the system GCC assembler/linker and
    # its extracted f951 and libgfortran development files.
    if compiler.is_absolute():
        lib=compiler.parent.parent/'lib/gcc/x86_64-linux-gnu/15'
        flags += ['-B'+str(lib)+'/', '-L'+str(lib), '-B/usr/lib/gcc/x86_64-linux-gnu/15/']
    tasks=[]
    for mode in args.variants:
        variant=routines
        if mode in ('dp_endpoint_fixed','dp_current','dp_current_ic_loss','dp_fixed_ic_loss'):
            assert variant.count('xintl=xgdpl*fbgpl/ebgm')==1
            variant=variant.replace('xintl=xgdpl*fbgpl/ebgm','xintl=xgdpl*fbgpl/ebgl')
        elif mode!='original': raise ValueError(mode)
        if mode in ('dp_current','dp_current_ic_loss'):
            start=variant.index('      subroutine dp_t_ele(')
            end=variant.index('      subroutine pc_cs_tot(',start)
            dp=variant[start:end]
            dp=dp.replace('      real*8 eele2              ! energy of the positron ( electron )',
                '      real*8 d,root,den,z1,z2\n      real*8 eele2              ! energy of the positron ( electron )')
            dp=dp.replace('if ((eele1.le.mele).or.(eele2.lt.mele)) then',
                'if ((eele1.le.mele).or.(eele2.le.0.d0)) then')
            a=dp.index('        ebgcuts=temp*1.0d-4')
            b=dp.index('        ebgl=ebgmin',a)
            dp=dp[:a]+"""        d=epho-eele1
        root=sqrt(eele1**2-mele**2)
        den=4.d0*epho*d+mele**2
        z1=epho*(mele**2+2.d0*d*mele**2/(root+eele1))/den
        z2=epho*(mele**2+2.d0*d*(root+eele1))/den
        ebgmin=max(mele**2/epho,z1)
        ebgmax=min(z2,epho,100.d0*temp)
        if (ebgmin.ge.ebgmax) then
          tdp=0.d0
          return
        endif
"""+dp[b:]
            dp=dp.replace('      real*8 g1,g2', '      real*8 delta,low,high,rad\n      real*8 g1,g2')
            # Only dp_fn depends on the integration variable ebg.
            fn=dp.index('      subroutine  dp_fn(')
            first,second=dp[:fn],dp[fn:]
            second=second.replace('eele2=epho-eele1','eele2=epho+ebg-eele1')
            second=second.replace('apho=epho/mele', 'apho=(epho+ebg)/mele')
            guard="""      if (epho*ebg.lt.mele**2) then
        xgdp=0.d0
        return
      endif
      rad=max(1.d0-mele**2/(epho*ebg),0.d0)
      delta=(epho-ebg)*sqrt(rad)
      low=(epho+ebg-delta)/2.d0
      high=(epho+ebg+delta)/2.d0
      if (eele1.lt.low.or.eele1.gt.high) then
        xgdp=0.d0
        return
      endif
"""
            second=second.replace('      if ((eele1.gt.mele)',guard+'      if ((eele1.gt.mele)')
            second=second.replace('      return\n      end', '      xgdp=max(xgdp,0.d0)\n      return\n      end')
            variant=variant[:start]+first+second+variant[end:]

        if mode in ('dp_current_ic_loss','dp_fixed_ic_loss'):
            assert variant.count('ephoub=eele-mele')==1
            variant=variant.replace('ephoub=eele-mele','ephoub=eele')
        source=out/f'{mode}.f'; source.write_text(driver+variant)
        executable=out/mode
        subprocess.run([str(compiler),*flags,str(source),'-o',str(executable)],check=True)
        for t in args.temperatures:
            tasks.append((str(executable),t,str(out/f'{mode}_T{t:.0e}')))
    (out/('build_metadata_'+'_'.join(args.variants)+'.json')).write_text(json.dumps(dict(original_sha256=hashlib.sha256((ROOT/'moroi_old.f').read_bytes()).hexdigest(),compiler=str(compiler),flags=flags,temperatures_GeV=args.temperatures,variants=args.variants),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=min(6,len(tasks))) as pool:
        for future in as_completed([pool.submit(run,task) for task in tasks]):
            print(future.result(),flush=True)


if __name__=='__main__': main()
