from fastapi import HTTPException
from src.models import InspectionNote,SegmentationRun
from src.utils.csv_export import csv_response
from src.services.audit import record

MODE_LABELS={'free':'自由分段','monotone_mean':'均温单调不降约束'}

def export_note(db,id,actor):
 note=db.get(InspectionNote,id)
 if not note:raise HTTPException(404,'说明不存在')
 if note.status!='approved':raise HTTPException(409,'仅已确认巡查说明可导出')
 run=db.get(SegmentationRun,note.run_id);result=run.result if run else {}
 mode=(result or {}).get('mode',(run.config if run else {}).get('mode','free'))
 rows=[[note.name,run.name,MODE_LABELS.get(mode,mode),run.result['fingerprint'],s['position'],s['start']+1,s['end'],s['mean'],s['rss'],note.interpretation,note.next_step,note.review_note] for s in note.segment_snapshot]
 record(db,actor,'note.exported',{'note_id':id,'mode':mode});return csv_response(f'thermal-inspection-{id}.csv',rows,['巡查说明','运行','分段方式','输入指纹','段号','起始序号(含)','结束序号(含)','均温','RSS','人工解释','下一步','审核意见'])
