from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import transfer as t


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path.cwd()))
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bundle=self.root/'bundle'; self.bundle.mkdir()
        self.logs=self.root/'logs'; self.logs.mkdir()
        self.sensors=self.root/'sensors'; self.sensors.mkdir()
        self.out=self.root/'snapshot'
        self.meta=dict(token='abcdef',session_id='session1',role='train',origin='legacy')
        selection=self.root/'selection.json'
        t.write(selection,dict(records=[dict(self.meta,log_name='log')]))
        t.write(self.bundle/'identity.json',dict(counts=dict(scenes=1,sessions=1),protected_sessions=['reserved'],
                                                inputs={str(selection):t.sha(selection)}))
        example=self.bundle/'examples/abcdef.npz'; example.parent.mkdir(); example.write_bytes(b'feature-and-label-snapshot')
        t.write(self.bundle/'examples_index.json',dict(abcdef=dict(metadata=self.meta,sha256=t.sha(example))))
        (self.bundle/'vocab.npz').write_bytes(b'vocab')
        t.write(self.bundle/'bundle_manifest.json',{k:t.sha(self.bundle/n) for n,k in
            (('identity.json','identity_sha256'),('examples_index.json','index_sha256'),('vocab.npz','vocab_sha256'))})
        frames=[]
        for i in range(4):
            cams={}
            for k in t.CAMERAS:
                name=f'log/{i}-{k}.jpg'
                path=t.checked(self.sensors,name); path.parent.mkdir(exist_ok=True)
                path.write_bytes(bytes([i])*64)
                cams[k.upper()]=dict(data_path=name,intrinsics=[[1,0,0],[0,1,0],[0,0,1]])
            frames.append(dict(token='abcdef' if i==3 else f'past{i}',cams=cams))
        with (self.logs/'log.pkl').open('wb') as f: pickle.dump(frames,f)
        self.capture=redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.addCleanup(self.capture.__exit__,None,None,None)

    def export(self): return t.export(self.bundle,self.logs,self.sensors,self.out)

    def test_snapshot_contains_only_selected_train_and_verifies_after_relocation(self):
        info=self.export()
        package=t.read(self.out/'package_manifest.json')
        self.assertEqual(package['images'],32)
        self.assertEqual(package['holdout_scenes'],0)
        self.assertFalse(any('.pkl' in name for name in package['files']))
        scenes=t.read(self.out/'scene_inputs.json')
        self.assertEqual(scenes[0]['metadata'],self.meta)
        self.assertEqual(len(scenes[0]['history']),4)
        self.assertEqual(t.extract(Path(info['archive']),self.root/'relocated',info['manifest_sha256'])['scenes'],1)
        self.assertEqual(self.export(),info)

    def test_resume_prefix_and_end_to_end_checksums_without_ssh_in_tests(self):
        info=self.export()
        archive=Path(info['archive'])
        dest=self.root/'dest'; dest.mkdir()
        part=dest/'training.tar.part'; part.write_bytes(archive.read_bytes()[:10001])
        def local(host,port,control,container,code): return [sys.executable,'-c',code]
        with patch.object(t,'remote_command',side_effect=local):
            result=t.pull('host',22,Path('socket'),'container',archive,dest,1)
        self.assertEqual(result['status'],'verified')
        self.assertEqual(t.sha(dest/'training.tar'),info['sha256'])
        self.assertFalse(part.exists())

    def test_corrupt_prefix_preserved_and_not_appended(self):
        info=self.export()
        dest=self.root/'dest'; dest.mkdir()
        part=dest/'training.tar.part'; part.write_bytes(b'corrupt')
        with patch.object(t,'remote_command',side_effect=lambda h,p,s,c,code:[sys.executable,'-c',code]):
            with self.assertRaisesRegex(RuntimeError,'prefix checksum'):
                t.pull('h',22,Path('s'),'c',Path(info['archive']),dest,1)
        self.assertEqual(part.read_bytes(),b'corrupt')

    def test_protected_session_refused(self):
        identity=t.read(self.bundle/'identity.json'); identity['protected_sessions']=['session1']
        t.write(self.bundle/'identity.json',identity)
        manifest=t.read(self.bundle/'bundle_manifest.json'); manifest['identity_sha256']=t.sha(self.bundle/'identity.json')
        t.write(self.bundle/'bundle_manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'unprotected training'):
            self.export()

    def test_source_image_changed_after_freeze_refuses_resume(self):
        self.export()
        image=next(self.sensors.rglob('*.jpg'))
        image.write_bytes(b'changed-image')
        with self.assertRaisesRegex(RuntimeError,'Snapshot inputs changed'):
            self.export()

    def test_archive_rejects_traversal_and_links(self):
        for name,link in (('../escape',False),('link',True)):
            archive=self.root/'evil.tar'
            manifest=dict(files={name:dict(size=0,sha256='x')})
            raw=(__import__('json').dumps(manifest)).encode()
            with tarfile.open(archive,'w') as tar:
                m=tarfile.TarInfo('package_manifest.json'); m.size=len(raw); tar.addfile(m,io.BytesIO(raw))
                m=tarfile.TarInfo(name)
                if link: m.type=tarfile.SYMTYPE; m.linkname='/tmp/escape'
                tar.addfile(m,io.BytesIO(b''))
            expected=__import__('hashlib').sha256(raw).hexdigest()
            with self.assertRaises(ValueError):
                t.extract(archive,self.root/'bad',expected)


if __name__=='__main__': unittest.main()
