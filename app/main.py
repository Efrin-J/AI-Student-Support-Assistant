from __future__ import annotations
import hashlib, json, os, re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = cosine_similarity = None

ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(ROOT.parent / '.env')
DATA = ROOT / 'data'
HISTORY, PROFILES = DATA / 'history', DATA / 'profiles'
KB_PATH, TICKETS_PATH = DATA / 'kb.json', DATA / 'tickets.json'
HISTORY.mkdir(parents=True, exist_ok=True); PROFILES.mkdir(parents=True, exist_ok=True)
if not TICKETS_PATH.exists(): TICKETS_PATH.write_text('[]', encoding='utf-8')

app = FastAPI(title='Campus Assist', version='3.0.0', description='AI Student Support Assistant — Groq + RAG + Tools + Memory')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class ChatIn(BaseModel):
    student_id: str = Field(min_length=3, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
class ProfileIn(BaseModel):
    name: str = ''; roll_no: str = ''; department: str = ''; semester: str = ''
class KBIn(BaseModel):
    category: str; title: str; content: str

STOP=set('the a an is are of to for in on and my i do how what does can please about me this that your with from tell give show was were be it its you any there their have has had am as at by or if who when where which will would should could may might into than then them they we our us'.split())
GENERAL={'hi','hello','hey','good morning','good afternoon','good evening','thanks','thank you','thx','bye','goodbye','are you ai','are you an ai','who are you','what are you','what is ai'}
COLLEGE={'attendance','attendence','syllabus','course','regulation','policy','rule','exam','examination','semester','grade','grading','backlog','supplementary','re-exam','reexam','condonation','detention','fee','fees','notice','noticeboard','hostel','library','transcript','hall ticket','elective','placement','college','campus','department','academic','admission','id card','student','ticket','complaint','support','cs201','cs305','ma202'}

def load(path, fallback):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except Exception:return fallback
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sid_ok(s): return re.fullmatch(r'[A-Za-z0-9_-]{3,100}',s or '') is not None
def docs(): return load(KB_PATH,[])
def profile(sid): return load(PROFILES/f'{sid}.json',{'name':'','roll_no':'','department':'','semester':''})
def history(sid): return load(HISTORY/f'{sid}.json',[])
def toks(s): return [x for x in re.findall(r'[a-z0-9]+',s.lower()) if x not in STOP and len(x)>1]
def course(q):
    m=re.search(r'\b([A-Za-z]{2,4})\s?(\d{3})\b',q); return (m.group(1)+m.group(2)).upper() if m else None

def category(q):
    q=q.lower()
    if any(x in q for x in ['notice','announcement','recent','latest','circular']): return 'Notice'
    if any(x in q for x in ['syllabus','units','textbook','course content','course outline']): return 'Syllabus'
    if any(x in q for x in ['faq','how do i get','where can i get']): return 'FAQ'
    if any(x in q for x in ['attendance','regulation','policy','rule','eligible','re-exam','reexam','supplementary','fee','grading','conduct','leave']): return 'Regulation'
    return None

def general(q):
    low=re.sub(r'\s+',' ',q.strip().lower()).strip(' .!?')
    if low in GENERAL or any(x in low for x in ['are you ai','are u ai','who are you','what are you','what can you do','what is your purpose']): return True
    return len(toks(q))<=2 and not course(q) and not any(x in low for x in COLLEGE)

def route(q):
    low=q.lower()
    human=any(x in low for x in ['human follow up','human follow-up','talk to a person','speak to someone','contact a human','support ticket','raise a complaint','contact support','human support'])
    if human:
        concrete=any(x in low for x in ['because','issue','problem','not working',"isn't",'isnt','cannot',"can't",'cant','unable','missing','failed','error','showing']) or len(toks(q))>=8
        return 'support' if concrete else 'support_clarify'
    if any(x in low for x in ['remember my name','remember my details','what do you remember','my profile','do you remember me']): return 'memory'
    if general(q): return 'general'
    if course(q) or any(x in low for x in COLLEGE): return 'college'
    return 'general'

def search(q, cat=None, limit=5, code=None):
    ds=docs(); filtered=[d for d in ds if not cat or d.get('category','').lower()==cat.lower()] or ds
    qt=toks(q); low=q.lower(); corpus=[]
    for d in filtered:
        corpus.append(' '.join([d.get('title',''),d.get('category',''),d.get('content',''),' '.join(d.get('keywords',[])),str(d.get('course',''))]))
    sem=[0.0]*len(filtered)
    if TfidfVectorizer and cosine_similarity and len(filtered)>1:
        try:
            v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True); m=v.fit_transform(corpus+[q]); sem=cosine_similarity(m[-1],m[:-1]).ravel().tolist()
        except Exception: pass
    scored=[]
    for i,d in enumerate(filtered):
        title=d.get('title','').lower(); text=corpus[i].lower(); keys=[str(x).lower() for x in d.get('keywords',[])]
        lex=sum((8 if re.search(rf'\b{re.escape(w)}\b',title) else 3 if re.search(rf'\b{re.escape(w)}\b',text) else 0)+(5 if w in keys else 0) for w in qt)
        exact=0
        if code:
            if d.get('id','').lower()==f'syl-{code.lower()}': exact+=180
            if str(d.get('course','')).upper()==code: exact+=100
            if code.lower() in text: exact+=45
        boost=0
        for topic,terms in {'attendance':['attendance'],'syllabus':['syllabus','course'],'fee':['fee','payment'],'exam':['exam','examination'],'supplementary':['supplementary','re-examination'],'hostel':['hostel'],'transcript':['transcript'],'library':['library'],'hall ticket':['hall ticket'],'elective':['elective'],'placement':['placement']}.items():
            if topic in low and any(t in title for t in terms): boost+=85
        if cat and d.get('category','').lower()==cat.lower(): boost+=12
        if d.get('date') and any(x in low for x in ['recent','latest','this week','notice']): boost+=4
        score=sem[i]*45+lex+exact+boost
        if score>0: scored.append((score,sem[i],d))
    scored.sort(key=lambda x:(x[0],x[1],x[2].get('date','')),reverse=True)
    if not scored or (scored[0][0]<4 and scored[0][1]<.08): return []
    cutoff=max(4,scored[0][0]*(.68 if len(qt)<=2 else .52))
    return [d for s,_,d in scored[:limit] if s>=cutoff]

def seed(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def attendance(sid,code):
    pct=55+seed((profile(sid).get('roll_no') or sid)+code)%40
    return {'course_code':code,'attendance_percent':pct,'status':'eligible for exam' if pct>=75 else 'eligible only with condonation' if pct>=65 else 'detained — below minimum threshold'}
def exam(code):
    n=seed(code)%9; return {'course_code':code,'exam_date':f'2026-09-{22+n:02d}','session':'Morning (9:30 AM)' if n%2==0 else 'Afternoon (2:00 PM)','known_course':any(d.get('id')=='syl-'+code.lower() for d in docs())}
def ticket(sid,cat,subject,description):
    items=load(TICKETS_PATH,[]); item={'id':f'TCK-{1001+len(items)}','category':cat,'subject':subject,'description':description,'student':sid,'status':'open','created':date.today().isoformat()}; items.append(item); save(TICKETS_PATH,items); return item

def tool(name,sid,inp):
    if name=='search_knowledge_base': return {'found':bool(search(inp['query'],inp.get('category'),5,inp.get('course_code'))),'documents':search(inp['query'],inp.get('category'),5,inp.get('course_code'))}
    if name=='check_attendance': return attendance(sid,inp['course_code'])
    if name=='get_exam_schedule': return exam(inp['course_code'])
    if name=='raise_ticket': return ticket(sid,inp['category'],inp['subject'],inp['description'])
    if name=='get_student_profile': return profile(sid)
    if name=='update_student_profile':
        p=profile(sid); p[inp['field']]=inp['value']; save(PROFILES/f'{sid}.json',p); return {'updated':inp['field'],'value':inp['value']}
    raise ValueError('Unknown tool')

SCHEMAS=[{'type':'function','function':{'name':'search_knowledge_base','description':'Search official college regulations, syllabus, FAQs and notices. Never invent college rules when this tool has not been consulted.','parameters':{'type':'object','properties':{'query':{'type':'string'},'category':{'type':'string','enum':['Regulation','Syllabus','FAQ','Notice','']},'course_code':{'type':'string'}},'required':['query']}}},{'type':'function','function':{'name':'check_attendance','description':'Check demo student attendance for a course.','parameters':{'type':'object','properties':{'course_code':{'type':'string'}},'required':['course_code']}}},{'type':'function','function':{'name':'get_exam_schedule','description':'Get demo exam date and session for a course.','parameters':{'type':'object','properties':{'course_code':{'type':'string'}},'required':['course_code']}}},{'type':'function','function':{'name':'raise_ticket','description':'Create a support request when the student asks for human follow-up and has described a concrete issue.','parameters':{'type':'object','properties':{'category':{'type':'string'},'subject':{'type':'string'},'description':{'type':'string'}},'required':['category','subject','description']}}},{'type':'function','function':{'name':'get_student_profile','description':'Retrieve saved student profile when asked what is remembered.','parameters':{'type':'object','properties':{},'required':[]}}},{'type':'function','function':{'name':'update_student_profile','description':'Remember a student-provided profile fact.','parameters':{'type':'object','properties':{'field':{'type':'string','enum':['name','roll_no','department','semester']},'value':{'type':'string'}},'required':['field','value']}}}]

def enabled(): return bool(os.getenv('GROQ_API_KEY')) and Groq is not None
def llm_history(sid): return [{'role':x['role'],'content':x['content']} for x in history(sid)[-12:] if x.get('role') in {'user','assistant'} and x.get('content')]

def groq_answer(sid,q):
    client=Groq(api_key=os.environ['GROQ_API_KEY']); model=os.getenv('GROQ_MODEL','openai/gpt-oss-120b'); p=profile(sid)
    system=f'''You are Campus Assist, an AI student-support assistant. Answer using official campus records and local tools. Never invent regulations, dates, fees, eligibility rules, or student facts. Use search_knowledge_base for college-specific information; attendance/exam tools for course data; get_student_profile for memory; update_student_profile only for explicit profile facts. Use raise_ticket for human follow-up when the message contains a concrete issue, extracting a concise subject and description. If the user asks for human help without describing an issue, do not create a ticket; ask them to describe it. For greetings, casual conversation, identity/purpose questions and general knowledge, answer naturally without campus tools. If a college question has no supporting record, say so. Be concise. Mention source document titles when using records. Current profile: {json.dumps(p,ensure_ascii=False)}'''
    messages=[{'role':'system','content':system}]+llm_history(sid)+[{'role':'user','content':q}]; trace=[]
    for _ in range(4):
        r=client.chat.completions.create(model=model,messages=messages,temperature=.2,max_completion_tokens=1200,tools=SCHEMAS,tool_choice='auto'); msg=r.choices[0].message
        if not msg.tool_calls: return msg.content or 'I could not produce a reliable answer.',trace,model
        messages.append(msg)
        for c in msg.tool_calls:
            try:
                args=json.loads(c.function.arguments or '{}'); result=tool(c.function.name,sid,args); trace.append({'tool':c.function.name,'input':args,'result':result}); content=json.dumps(result,ensure_ascii=False)
            except Exception as e: trace.append({'tool':c.function.name,'error':str(e)}); content=json.dumps({'error':str(e)})
            messages.append({'role':'tool','tool_call_id':c.id,'name':c.function.name,'content':content})
    return 'I reached the tool-use limit. Please try a more specific question.',trace,model

def local_answer(sid,q):
    r=route(q)
    if r=='support_clarify': return 'Sure. Tell me the issue you want a human staff member to follow up on, and I can create a support request.',[],None,'router-support-clarify'
    if r=='general': return 'I’m Campus Assist, an AI student-support assistant. I can help with college regulations, syllabus, FAQs, notices, student records/services, and support requests. Ask me a question.',[],None,'local-general'
    if r=='memory': return f"Here is what I currently remember: {json.dumps(profile(sid),ensure_ascii=False)}",[],None,'local-memory'
    code=course(q); low=q.lower()
    if code and 'attendance' in low: return json.dumps(attendance(sid,code)),[{'tool':'check_attendance','result':attendance(sid,code)}],None,'local-tool'
    if code and ('exam' in low or 'examination' in low) and any(x in low for x in ['when','date','schedule']): return json.dumps(exam(code)),[{'tool':'get_exam_schedule','result':exam(code)}],None,'local-tool'
    cat=category(q); ds=search(q,cat,5,code)
    if not ds: return 'I could not find a supporting record in the current campus knowledge base. Please check with the relevant college office.',[],None,'local-rag'
    text='\n\n'.join(f"{d['title']}: {d['content']}" for d in ds[:3]); return text,[{'tool':'search_knowledge_base','input':{'query':q,'category':cat,'course_code':code},'result':{'found':True,'documents':ds}}],None,'local-rag'

@app.get('/')
def root(): return FileResponse(ROOT/'static/index.html')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
@app.get('/api/health')
def health(): return {'status':'ok','service':'Campus Assist','mode':'groq-rag-tools' if enabled() else 'local-rag-tools','groq_configured':enabled()}
@app.get('/api/kb')
def get_kb(): return docs()
@app.post('/api/kb')
def add_kb(body:KBIn):
    if body.category not in {'Regulation','Syllabus','FAQ','Notice'}: raise HTTPException(400,'Invalid category')
    ds=docs(); item={'id':'custom-'+datetime.now().strftime('%Y%m%d%H%M%S%f'),'category':body.category,'title':body.title.strip(),'content':body.content.strip(),'date':date.today().isoformat()}; ds.append(item); save(KB_PATH,ds); return item
@app.get('/api/profile/{sid}')
def get_profile(sid:str):
    if not sid_ok(sid): raise HTTPException(400,'Invalid student id')
    return profile(sid)
@app.post('/api/profile/{sid}')
def save_profile(sid:str,body:ProfileIn):
    if not sid_ok(sid): raise HTTPException(400,'Invalid student id')
    p=body.model_dump(); save(PROFILES/f'{sid}.json',p); return p
@app.post('/api/chat')
def chat(body:ChatIn):
    if not sid_ok(body.student_id): raise HTTPException(400,'Invalid student id')
    if enabled():
        r=route(body.message)
        if r=='support_clarify': reply,trace,model,mode=local_answer(body.student_id,body.message)
        else:
            try: reply,trace,model=groq_answer(body.student_id,body.message); mode='groq-agent'
            except Exception as e: reply,trace,model=local_answer(body.student_id,body.message); mode='local-fallback'
    else: reply,trace,model,mode=local_answer(body.student_id,body.message)
    h=history(body.student_id); h.extend([{'role':'user','content':body.message,'at':datetime.now().isoformat()},{'role':'assistant','content':reply,'at':datetime.now().isoformat()}]); save(HISTORY/f'{body.student_id}.json',h[-100:])
    return {'reply':reply,'toolTrace':trace,'model':model,'mode':mode}
