"""Optional retrieval of public day 111, with the original panel rule."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import zipfile

import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from retrieve_hourly import BASE, Remote

ARCHIVE_BYTES = 351803513445
TABLE = 'asi_opensource_pod_hourly'
URL = BASE + TABLE + '.zip'
COLUMNS = ['pod_id', 'cluster_id', 'state_public', 'priority_class',
           'job_type_public', 'gpu_request', 'used_gpu_hours']
DAYS = (111,)
DESTINATION = HERE / 'additional_trace_partitions'


def fetch_one(day, hour, entry, workers):
    path = DESTINATION / f'day{day}_hour{hour:02d}.parquet'
    manifest = path.with_suffix('.json')
    if path.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']:
            return record
    info = zipfile.ZipInfo(entry['name'])
    info.header_offset, info.file_size = entry['offset'], entry['size']
    info.compress_type = entry['compression']
    source = Remote(URL, ARCHIVE_BYTES).member(info)
    source.seek(source.size-8)
    length = struct.unpack('<I', source.read(8)[:4])[0]
    source.seek(source.size-length-8)
    footer = source.read(length+8)
    with TemporaryDirectory(prefix='trace-range-') as tmp:
        sparse = Path(tmp) / 'partition.parquet'
        with sparse.open('wb') as writer:
            writer.write(b'PAR1')
            writer.truncate(source.size)
            writer.seek(source.size-len(footer))
            writer.write(footer)
        metadata = pq.ParquetFile(sparse).metadata
        blocks = []
        for row in range(metadata.num_row_groups):
            group = metadata.row_group(row)
            ranges = []
            for i in range(group.num_columns):
                column = group.column(i)
                if column.path_in_schema in COLUMNS:
                    start = min(q for q in (column.dictionary_page_offset, column.data_page_offset)
                                if q is not None and q >= 0)
                    ranges.append((start, start+column.total_compressed_size))
            merged = []
            for start, end in sorted(ranges):
                if merged and start-merged[-1][1] < 16384:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            blocks.extend(merged)

        def get_block(block):
            start, end = block
            return start, Remote(URL, end-start, source.offset+start).read(end-start)

        tasks = [workers.submit(get_block, block) for block in blocks]
        with sparse.open('r+b') as writer:
            for task in as_completed(tasks):
                offset, data = task.result()
                writer.seek(offset)
                writer.write(data)
        frame = pq.read_table(sparse, columns=COLUMNS)
        mask = pc.and_(pc.starts_with(frame['pod_id'], '00'), pc.greater(frame['gpu_request'], 0))
        filtered = frame.filter(mask)
        pending = path.with_suffix('.pending')
        pq.write_table(filtered, pending, compression='zstd')
        pending.replace(path)
    record = dict(day=day, hour=hour, member=entry['name'], archive_offset=entry['offset'],
                  source_rows=metadata.num_rows, selected_rows=filtered.num_rows,
                  source_bytes=sum(b-a for a,b in blocks), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    print(day, hour, record['selected_rows'], flush=True)
    return record


def main():
    DESTINATION.mkdir(exist_ok=True)
    archive = HERE.parent / 'hourly_trace/source_partitions/archive_members.json'
    index = {x['name']:x for x in json.loads(archive.read_text())}
    selection = [(d,h,index[f'{TABLE}/day={d}/hour={h:02d}/part-000.parquet'])
                 for d in DAYS for h in range(24)]
    with ThreadPoolExecutor(max_workers=24) as readers, ThreadPoolExecutor(max_workers=6) as hours:
        list(hours.map(lambda item: fetch_one(*item, readers), selection))


if __name__ == '__main__':
    main()
