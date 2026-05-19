import streamlit as st
import spacy
import language_tool_python
import re
import requests
import json
import feedparser
from openai import OpenAI
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Video Skript Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Video Skript Analyzer")
st.markdown("Analysiere deine Videoskripte auf Rechtschreibung, Style & Fakten + News-Quellen")

# ============ API-KEY VERWALTUNG ============

def get_api_keys():
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        
        openai_key = st.text_input(
            "OpenAI API-Key:",
            type="password",
            help="Für ChatGPT Faktenchecking"
        )
        
        newsapi_key = st.text_input(
            "NewsAPI Key (optional):",
            type="password",
            help="Für News-Quellen (von newsapi.org)"
        )
        
        if openai_key:
            st.success("✅ OpenAI aktiviert")
        else:
            st.warning("⚠️ OpenAI erforderlich")
        
        if newsapi_key:
            st.success("✅ NewsAPI aktiviert")
        else:
            st.info("ℹ️ NewsAPI optional")
        
        return openai_key, newsapi_key

openai_key, newsapi_key = get_api_keys()

# ============ ANALYZER CLASS ============

class VideoScriptAnalyzer:
    def __init__(self, openai_key: str, newsapi_key: str = None):
        self.nlp = spacy.load("de_core_news_sm")
        self.grammar_tool = language_tool_python.LanguageTool('de-DE')
        self.client = OpenAI(api_key=openai_key)
        self.newsapi_key = newsapi_key
        
        # RSS Feeds für News
        self.rss_sources = {
            'Tagesschau': 'https://www.tagesschau.de/xml/rss2_de/',
            'DW': 'https://www.dw.com/de/feed/rss',
            'ARD': 'https://www.ard.de/feed',
        }
        
        self.ignore_patterns = [
            "Außer am Satzanfang",
            "Leerzeichen zu viel",
            "Hier scheint",
            "Wiederholung",
        ]
    
    def _should_ignore_error(self, error_message: str, word: str) -> bool:
        for pattern in self.ignore_patterns:
            if pattern.lower() in error_message.lower():
                return True
        
        if "klein" in error_message.lower() and word and word[0].isupper():
            return True
        
        if len(word) < 2:
            return True
        
        return False
    
    def analyze_spelling_grammar(self, text: str) -> dict:
        matches = self.grammar_tool.check(text)
        
        errors = []
        for match in matches:
            word = match.word if hasattr(match, 'word') else ''
            
            if self._should_ignore_error(match.message, word):
                continue
            
            if match.replacements:
                errors.append({
                    'message': match.message,
                    'suggestion': match.replacements[:2],
                    'word': word,
                })
        
        return {
            'total_errors': len(errors),
            'errors': errors,
            'score': max(0, 100 - (len(errors) * 5))
        }
    
    def extract_facts(self, text: str) -> list:
        doc = self.nlp(text)
        facts = []
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            entities = [(ent.text, ent.label_) for ent in sent.ents]
            
            has_numbers = any(token.like_num for token in sent)
            has_entities = len(entities) > 0
            
            if has_numbers or has_entities:
                facts.append({
                    'text': sentence_text,
                    'entities': entities,
                    'has_numbers': has_numbers
                })
        
        return facts
    
    def verify_facts(self, facts: list) -> dict:
        verified = []
        unverified = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fact in enumerate(facts):
            status_text.text(f"🔍 Überprüfe {i+1}/{len(facts)}...")
            result = self._check_fact_with_multiple_sources(fact)
            
            if result['status'] == 'verified':
                verified.append(result)
            else:
                unverified.append(result)
            
            progress_bar.progress((i + 1) / len(facts))
        
        status_text.empty()
        
        return {
            'total_facts': len(facts),
            'verified': verified,
            'unverified': unverified,
            'verification_rate': round(len(verified) / len(facts) * 100, 1) if facts else 0
        }
    
    def _check_fact_with_multiple_sources(self, fact: dict) -> dict:
        """Checkt mit mehreren Quellen"""
        text = fact['text']
        
        # 1. NewsAPI
        if self.newsapi_key:
            news_result = self._search_newsapi(text)
            if news_result['found']:
                return {
                    'text': text[:80],
                    'status': 'verified',
                    'reason': f"📰 News: {news_result['title'][:50]}",
                    'confidence': 0.9,
                    'source': news_result['source'],
                    'url': news_result['url']
                }
        
        # 2. RSS-Feeds
        rss_result = self._search_rss_feeds(text)
        if rss_result['found']:
            return {
                'text': text[:80],
                'status': 'verified',
                'reason': f"📻 {rss_result['source']}: {rss_result['title'][:40]}",
                'confidence': 0.85,
                'source': rss_result['source'],
                'url': rss_result['url']
            }
        
        # 3. ChatGPT
        gpt_result = self._check_fact_with_gpt(fact)
        
        if gpt_result['status'] == 'verified' and gpt_result['confidence'] > 0.8:
            return gpt_result
        else:
            return {
                'text': text[:80],
                'status': 'unverified',
                'reason': gpt_result['reason'],
                'confidence': gpt_result['confidence'],
                'source': '🤖 ChatGPT'
            }
    
    def _search_newsapi(self, text: str) -> dict:
        """Sucht in NewsAPI"""
        try:
            keywords = text.split()[:3]
            search_term = ' '.join(keywords)
            
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params={
                    'q': search_term,
                    'language': 'de',
                    'sortBy': 'relevancy',
                    'pageSize': 5,
                    'apiKey': self.newsapi_key
                },
                timeout=5
            )
            
            articles = response.json().get('articles', [])
            if articles:
                article = articles[0]
                return {
                    'found': True,
                    'title': article['title'],
                    'source': article['source']['name'],
                    'url': article['url'],
                    'date': article['publishedAt']
                }
        except:
            pass
        
        return {'found': False}
    
    def _search_rss_feeds(self, text: str) -> dict:
        """Sucht in RSS Feeds"""
        try:
            keywords = text.split()[:2]
            search_term = ' '.join(keywords).lower()
            
            for source_name, rss_url in self.rss_sources.items():
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:20]:
                    entry_text = (
                        entry.get('title', '') + ' ' + 
                        entry.get('summary', '')
                    ).lower()
                    
                    if any(keyword.lower() in entry_text for keyword in keywords):
                        return {
                            'found': True,
                            'title': entry.get('title', 'Unknown')[:60],
                            'source': source_name,
                            'url': entry.get('link', '#'),
                            'date': entry.get('published', '')
                        }
        except:
            pass
        
        return {'found': False}
    
    def _check_fact_with_gpt(self, fact: dict) -> dict:
        """Überprüft mit ChatGPT"""
        text = fact['text']
        
        try:
            prompt = f"""Überprüfe diese Aussage auf Korrektheit:
"{text}"

Antworte mit JSON:
{{"is_true": true/false, "confidence": 0.0-1.0, "explanation": "kurz"}}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Du bist ein Faktenchecker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            result_data = json.loads(json_str)
            
            is_true = result_data.get('is_true', False)
            confidence = result_data.get('confidence', 0.5)
            explanation = result_data.get('explanation', '')
            
            if is_true and confidence > 0.75:
                return {
                    'status': 'verified',
                    'reason': explanation,
                    'confidence': confidence
                }
            else:
                return {
                    'status': 'unverified',
                    'reason': explanation,
                    'confidence': confidence
                }
        except:
            return {
                'status': 'error',
                'reason': 'ChatGPT Error',
                'confidence': 0
            }
    
    def check_buzzroom(self, text: str) -> dict:
        doc = self.nlp(text)
        issues = []
        score = 100
        
        for sent in doc.sents:
            words = len([t for t in sent if t.is_alpha])
            if words > 15:
                issues.append(f"⚠️ Satz zu lang ({words} Wörter)")
                score -= 2
        
        complex_words = ['infolgedessen', 'nichtsdestotrotz', 'demzufolge']
        for word in complex_words:
            if word in text.lower():
                issues.append(f"⚠️ Formales Wort: '{word}'")
                score -= 5
        
        return {'issues': issues, 'score': max(0, score)}
    
    def check_pressekodex(self, text: str) -> dict:
        issues = []
        score = 100
        
        sensational = ['unglaublich', 'sensationell', 'skandal']
        for word in sensational:
            if word in text.lower():
                issues.append(f"❌ Sensationalismus: '{word}'")
                score -= 10
        
        return {'issues': issues, 'score': max(0, score)}
    
    def analyze_full(self, text: str) -> dict:
        spelling = self.analyze_spelling_grammar(text)
        facts = self.extract_facts(text)
        fact_check = self.verify_facts(facts)
        buzzroom = self.check_buzzroom(text)
        pressekodex = self.check_pressekodex(text)
        
        overall = (
            spelling['score'] * 0.2 +
            buzzroom['score'] * 0.3 +
            pressekodex['score'] * 0.2 +
            fact_check['score'] * 0.3
        )
        
        return {
            'overall_score': round(overall, 1),
            'spelling': spelling,
            'buzzroom': buzzroom,
            'pressekodex': pressekodex,
            'facts': fact_check,
            'word_count': len(text.split())
        }

# ============ MAIN APP ============

if openai_key:
    st.header("📝 Skript eingeben")
    
    script_text = st.text_area(
        "Gib dein Videoskript ein:",
        height=200,
        placeholder="Dein Text hier..."
    )
    
    if st.button("🚀 Analysieren", type="primary", use_container_width=True):
        if not script_text:
            st.error("❌ Bitte gib einen Text ein!")
        else:
            with st.spinner("⏳ Analysiere..."):
                analyzer = VideoScriptAnalyzer(openai_key=openai_key, newsapi_key=newsapi_key)
                result = analyzer.analyze_full(script_text)
            
            st.success("✅ Fertig!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Gesamtscore", f"{result['overall_score']}/100")
            
            with col2:
                st.metric("Rechtschreibung", f"{result['spelling']['score']}/100")
            
            with col3:
                st.metric("Buzzroom", f"{result['buzzroom']['score']}/100")
            
            with col4:
                st.metric("Fakten", f"{result['facts']['score']}/100")
            
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Fakten", "✏️ Rechtschreibung", "🎬 Buzzroom", "📄 Pressekodex"])
            
            with tab1:
                st.subheader("Faktenchecking (News + ChatGPT)")
                st.info(f"**{result['facts']['total_facts']}** Aussagen - **{result['facts']['verification_rate']}%** verifiziert")
                
                col_v, col_u = st.columns(2)
                
                with col_v:
                    st.success(f"✅ {len(result['facts']['verified'])} Verifiziert")
                    for fact in result['facts']['verified'][:5]:
                        st.caption(f"• {fact['text']}")
                        if 'source' in fact:
                            st.caption(f"  Quelle: {fact['source']}")
                
                with col_u:
                    st.warning(f"❓ {len(result['facts']['unverified'])} Unverified")
                    for fact in result['facts']['unverified'][:5]:
                        st.caption(f"• {fact['text']}")
            
            with tab2:
                st.subheader("Rechtschreibung & Grammatik")
                
                if result['spelling']['errors']:
                    st.warning(f"**{result['spelling']['total_errors']}** Fehler:")
                    for error in result['spelling']['errors'][:10]:
                        st.write(f"• **{error['word']}** - {error['message']}")
                else:
                    st.success("✅ Keine Fehler!")
            
            with tab3:
                st.subheader("Buzzroom Guidelines")
                
                if result['buzzroom']['issues']:
                    for issue in result['buzzroom']['issues']:
                        st.warning(issue)
                else:
                    st.success("✅ OK!")
            
            with tab4:
                st.subheader("Pressekodex")
                
                if result['pressekodex']['issues']:
                    for issue in result['pressekodex']['issues']:
                        st.error(issue)
                else:
                    st.success("✅ OK!")

else:
    st.error("❌ OpenAI API-Key erforderlich!")
