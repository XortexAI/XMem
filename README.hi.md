<div align="center">
  <img
    src="https://github.com/user-attachments/assets/aa171a4c-074c-4082-b3d1-c70f5f7f2aca"
    alt="XMem Logo"
    width="100%"
  />
</div>

<div align="center">
  <h1>XMem</h1>
  <p><strong>AI के लिए मेमोरी लेयर जो कभी नहीं भूलती</strong></p>
  <p>हर AI एजेंट और LLM इंटरफेस को तुरंत स्थायी, क्रॉस-प्लेटफॉर्म मेमोरी प्रदान करें।</p>

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="BSD-3 License"/>
<img src="https://img.shields.io/badge/FastAPI-00C7B7?logo=fastapi&logoColor=white" alt="FastAPI"/>
<br/>
<img src="https://img.shields.io/badge/LangGraph-6C47FF?logo=langchain&logoColor=white" alt="LangGraph"/>
<img src="https://img.shields.io/badge/Rust-Weaver-b7410e?logo=rust&logoColor=white" alt="Rust Weaver"/>
<img src="https://img.shields.io/badge/Multi--LLM-Gemini%20%7C%20Claude%20%7C%20GPT%20%7C%20Bedrock%20%7C%20Ollama-orange" alt="Multi-LLM"/>
</div>

<hr>

<p align="center">
  <a href="README.md">English</a> &nbsp;&bull;&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;&bull;&nbsp;
  <a href="README.ja.md">日本語</a> &nbsp;&bull;&nbsp;
  <a href="README.hi.md">हिन्दी</a>
</p>

<p align="center">
  <a href="#डेमो">डेमो</a> &nbsp;&bull;&nbsp;
  <a href="#विशेषताएं">विशेषताएं</a> &nbsp;&bull;&nbsp;
  <a href="#आर्किटेक्चर">आर्किटेक्चर</a> &nbsp;&bull;&nbsp;
  <a href="#बेंचमार्क">बेंचमार्क</a> &nbsp;&bull;&nbsp;
  <a href="#त्वरित-शुरुआत">त्वरित शुरुआत</a> &nbsp;&bull;&nbsp;
  <a href="#कॉन्फ़िगरेशन">कॉन्फ़िगरेशन</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&theme=dark&legend=top-left" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
  <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
</picture>

## अपडेट / समाचार
- **[1 जून 2026]** XMem में अब इसकी मेमोरी लेयर का नेटिव Golang इंप्लीमेंटेशन है। उच्च थ्रूपुट, कम विलंबता और प्रोडक्शन-स्केल डिप्लॉयमेंट के लिए बनाया गया है जहां मेमोरी लाखों इंटरैक्शन में विश्वसनीयता से काम करती है।
- **[25 मई 2026]** लोकल वर्कस्पेस सपोर्ट अब लाइव है। बस 3 कमांड में XMem को लोकली सेटअप करें और मिनटों में मेमोरी के साथ बिल्डिंग शुरू करें। सेटअप निर्देशों के लिए [Local.md](https://github.com/XortexAI/XMem/blob/main/Local.md) देखें।
 ```bash
npx create-xmem@latest
cd xmem
npm run dev
```

## XMem क्या है?

LLM के साथ हर बातचीत शुरुआत से शुरू होती है। टूल बदलें, प्रोवाइडर बदलें, एक हफ्ते बाद वापस आएं और सारा कॉन्टेक्स्ट चला जाता है।

XMem भारत की #1 ओपन सोर्स Agentic मेमोरी लेयर है। हम Memory-as-a-Service पेश कर रहे हैं - हर AI यूज केस और डोमेन के लिए मेमोरी लेयर। चाहे यह लंबे समय तक चलने वाले एजेंटों के लिए टेम्पोरल मेमोरी हो, रोगी के संदर्भ के लिए मेडिकल मेमोरी हो, टीमों और प्रोजेक्ट्स के लिए एंटरप्राइज मेमोरी हो, या कोडिंग एजेंट्स और वर्कफ़्लो के लिए डेवलपर मेमोरी हो।

यह स्टेटफुल AI के लिए पहली तरह की agentic मेमोरी लेयर है।
परंपरागत मेमोरी सिस्टम के विपरीत जो सिर्फ चंक्स को स्टोर और रिट्रीव करते हैं, XMem मेमोरी को एक सक्रिय रीजनिंग प्रक्रिया में बदल देता है। यह तय करता है कि क्या याद रखना है, क्या अपडेट करना है, क्या भूलना है, और हर मेमोरी को सही स्पेशलाइज्ड एजेंट और स्टोर में डायनामिकली रूट करता है।

## डेमो

अपनी पसंद के किसी भी AI प्लेटफॉर्म पर बस "X" टाइप करें और XMem द्वारा दिए गए चार मोड्स में से चुनें ताकि आप अपनी मेमोरीज को सहजता से स्टोर और सर्च कर सकें, मौजूदा चैट्स से कॉन्टेक्स्ट इंपोर्ट कर सकें, या इंडेक्स किए गए रेपोज़ के साथ काम कर सकें।

https://github.com/user-attachments/assets/8e3349ab-63c9-4046-821d-ca8097948440

## विशेषताएं

### Chrome एक्सटेंशन

XMem Chrome एक्सटेंशन ChatGPT, Claude, Gemini, DeepSeek और Perplexity को स्थायी मेमोरी प्रदान करता है।

**लाइव सर्च और इंजेक्ट** - जैसे ही आप प्रॉम्प्ट टाइप करते हैं, XMem रीयल-टाइम में आपकी मेमोरी सर्च करता है और एक फ्लोटिंग चिप दिखाता है। एक क्लिक से प्रासंगिक कॉन्टेक्स्ट सीधे आपके इनपुट में इंजेक्ट हो जाता है, बिना किसी परेशानी के।

**बैकग्राउंड ऑटो-सेव (Xingest)** - जब आप "Send" दबाते हैं, XMem एसिंक्रोनसली बातचीत के टर्न को कैप्चर करता है। एक बैकग्राउंड कतार फैक्ट्स और सारांश निकालती है बिना आपके UI को छुए।

https://github.com/user-attachments/assets/97793cf9-d247-4d02-9c31-3cc9bbbf89aa

### एजेंट प्लगइन्स

नया [plugin/](plugin/) फोल्डर XMem को सीधे डेवलपर एजेंट्स और कोडिंग असिस्टेंट्स में लाता है। इसमें [Claude Code](plugin/xmem-claude/), [Codex](plugin/xmem-codex/), [Cursor](plugin/xmem-cursor/), [Hermes](plugin/xmem-hermes/), [OpenClaw](plugin/xmem-openclaw/), और [OpenCode](plugin/xmem-opencode/) के लिए फर्स्ट-पार्टी इंटीग्रेशन शामिल हैं, ताकि एजेंट्स मौजूदा मेमोरी को सर्च कर सकें, टिकाऊ प्रोजेक्ट ज्ञान सेव कर सकें, और API कीज़ को एनवायरनमेंट वेरिएबल्स या क्लाइंट-स्पेसिफिक सीक्रेट स्टोर्स में रखते हुए सेशन्स के बीच कॉन्टेक्स्ट ले जा सकें।

### कॉन्टेक्स्ट

कॉन्टेक्स्ट आपको मैनुअली कॉपी पेस्ट किए बिना मौजूदा बातचीत को XMem में लाने देता है।

एक शेयर्ड ChatGPT, Claude या Gemini लिंक पेस्ट करें। XMem इसे खोलता है, हर यूजर और असिस्टेंट मैसेज निकालता है, और पूरी इनजेस्ट पाइपलाइन चलाता है ताकि बातचीत सर्चेबल मेमोरी बन जाए।

आप एक ट्रांसक्रिप्ट फाइल (टेक्स्ट, मार्कडाउन या JSON) भी अपलोड कर सकते हैं। XMem में Cursor और Antigravity एक्सपोर्ट्स के लिए बिल्ट-इन पार्सिंग है और अनजान फॉर्मेट्स के लिए LLM फॉलबैक का उपयोग करता है।

https://github.com/user-attachments/assets/4ff22405-b7ad-4b78-9189-9a6e3ebd5e40

### स्कैनर

स्कैनर पूरे Git रिपोज़िटरीज़ को इंडेक्स करता है और आपके कोडबेस का एक क्वेरीएबल नॉलेज ग्राफ बनाता है।

एक बार इंडेक्स करने के बाद, आप फाइलों, फंक्शन्स, डिपेंडेंसीज़ और इम्पैक्ट के बारे में नेचुरल लैंग्वेज प्रश्न पूछ सकते हैं। इसका उपयोग एक नए रेपो को समझने के लिए, यह पता लगाने के लिए कि कोई फीचर कहाँ रहता है, कोड कैसे जुड़ा हुआ है यह देखने के लिए, या यह जानने के लिए करें कि यदि आप कुछ बदलते हैं तो क्या टूटेगा।

https://github.com/user-attachments/assets/f0fd393e-3820-404b-8d0e-e452e1dd52d0

### मल्टी-डोमेन क्लासिफिकेशन

सभी मेमोरीज़ समान नहीं हैं, और ऐसा मानना ही है कि अन्य समाधान खराब प्रदर्शन क्यों करते हैं। XMem का **Classifier एजेंट** आने वाले हर डेटा को विश्लेषण करता है और उसे सही डोमेन में रूट करता है:

<table>
  <tr>
    <th>डोमेन</th>
    <th>क्या स्टोर करता है</th>
    <th>उदाहरण</th>
    <th>स्टोरेज</th>
  </tr>
  <tr>
    <td><strong>प्रोफाइल</strong></td>
    <td>स्थायी यूजर फैक्ट्स, प्राथमिकताएं, पहचान</td>
    <td><em>"मैं बैकएंड्स के लिए Go को Python से बेहतर मानता हूँ"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>टेम्पोरल</strong></td>
    <td>समय-एंकर्ड इवेंट्स दिनांक रिज़ॉल्यूशन के साथ</td>
    <td><em>"मुझे कल Staff Engineer की प्रमोशन मिली"</em></td>
    <td>Neo4j</td>
  </tr>
  <tr>
    <td><strong>सारांश</strong></td>
    <td>कम्प्रेस्ड कनवरसेशन टेकअवे</td>
    <td><em>"REST से gRPC में माइग्रेशन पर चर्चा की"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>कोड</strong></td>
    <td>प्रतीकों से जुड़ी एनोटेशन्स, बग्स, व्याख्याएं</td>
    <td><em>"इस रिट्राई लॉजिक में एक race condition है"</em></td>
    <td>Neo4j + Pinecone</td>
  </tr>
  <tr>
    <td><strong>स्निपेट</strong></td>
    <td>व्यक्तिगत कोड पैटर्न्स और उपयोगिताएं</td>
    <td><em>"यहाँ Go में मेरा स्टैंडर्ड error handler है"</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>इमेज</strong></td>
    <td>विजुअल ऑब्जर्वेशन्स और विवरण</td>
    <td><em>आर्किटेक्चर डायग्राम का स्क्रीनशॉट</em></td>
    <td>Pinecone</td>
  </tr>
</table>

## आर्किटेक्चर

<img width="1536" height="1024" alt="WhatsApp Image 2026-04-27 at 11 50 51" src="https://github.com/user-attachments/assets/424d1c77-63e3-48ac-b457-6beecd437f65" />

XMem को LangGraph द्वारा कोऑर्डिनेट किए गए **स्पेशलाइज्ड AI एजेंट्स की पाइपलाइन** के रूप में बनाया गया है, जो एक डेटरमिनिस्टिक एक्सिक्यूशन लेयर (Weaver) और तीन पर्पस-बिल्ट स्टोरेज इंजन्स द्वारा समर्थित है।

### इनजेस्शन फ्लो

```
यूजर इनपुट (SDK / Chrome एक्सटेंशन / API)
         |
         v
   +--------------+
   |  Classifier   |    टेक्स्ट को विश्लेषण करता है, डोमेन्स को रूट करता है
   +------+-------+
          |
    +-----+-----+------+----------+
    v     v     v      v          v
 Profile Temporal Summary Code  Snippet     डोमेन एजेंट्स
 Agent   Agent   Agent  Agent   Agent       स्ट्रक्चर्ड डेटा निकालते हैं
    |     |      |      |        |          पैरेलल में
    v     v      v      v        v
   +----------------------------------+
   |          Judge एजेंट             |     मौजूदा मेमोरी से तुलना करता है
   |   (ADD / UPDATE / DELETE / NOOP) |     डुप्लिकेट्स और स्टेलनेस को रोकता है
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |        Weaver (Rust कोर)         |     डेटरमिनिस्टिक एक्सीक्यूटर
   |  Pinecone | Neo4j | MongoDB     |     कोई LLM नहीं। शुद्ध सॉफ्टवेयर लॉजिक।
   +----------------------------------+
```

### रिट्रीवल फ्लो

```
यूजर क्वेरी
    |
    v
+----------------------------------+
|       Retrieval LLM              |
|  तय करता है कि कौन सी टूल्स कॉल करें: |
|  SearchProfile, SearchTemporal,  |
|  SearchSummary, SearchSnippet    |
+----------------+-----------------+
                 |
    +------------+------------+
    v            v            v
 Pinecone      Neo4j      Pinecone        पैरेलल सर्च एक्सिक्यूशन
 (profiles)   (events)   (summaries)
    |            |            |
    +------------+------------+
                 v
+----------------------------------+
|   आंसर सिंथेसिस + साइटेशन्स     |    LLM सोर्स रेफरेंसेस के साथ जवाब जेनरेट करता है
+----------------------------------+
```

### स्टोरेज

<table>
  <tr>
    <th>इंजन</th>
    <th>उद्देश्य</th>
    <th>उपयोग किया जाता है</th>
  </tr>
  <tr>
    <td><strong>Pinecone</strong></td>
    <td>उच्च गति वेक्टर समानता सर्च</td>
    <td>प्रोफाइल्स, सारांश, स्निपेट्स, कोड एनोटेशन्स</td>
  </tr>
  <tr>
    <td><strong>Neo4j</strong></td>
    <td>ग्राफ ट्रैवर्सल + टेम्पोरल रीजनिंग</td>
    <td>इवेंट्स, कोड नॉलेज ग्राफ, एनोटेशन्स</td>
  </tr>
  <tr>
    <td><strong>MongoDB</strong></td>
    <td>कच्चा डॉक्यूमेंट स्टोरेज</td>
    <td>स्कैन किया गया कोड, फाइल मेटाडेटा, स्कैन स्टेट</td>
  </tr>
</table>

> [!नोट]
> लोकल डिप्लॉयमेंट्स के लिए, Pinecone को **Chroma**, **pgvector**, या **SQLite** वेक्टर स्टोर्स से बदला जा सकता है।

## त्वरित शुरुआत

### लोकल XMem

```bash
npx create-xmem@latest
cd xmem
npm run dev
```

यह Windows, macOS और Linux पर काम करता है। यह एक लोकल XMem वर्कस्पेस बनाता है, बैकएंड इंस्टॉल करता है, लोकल स्टोरेज शुरू करता है, Chrome एक्सटेंशन बिल्ड करता है, और API को `http://localhost:8000` पर लॉन्च करता है।

लोकल प्रीरिक्विजिट्स:

- Git
- Node.js 20+
- Python 3.11+
- Docker Desktop
- Ollama, जब तक कि आप `.env` में क्लाउड LLM की को न जोड़ें

सेटअप के बाद, एक्सटेंशन यहाँ से लोड करें:

```text
repos/xmem-extension/dist
```

Chrome पाथ: `chrome://extensions` -> Developer mode एनेबल करें -> Load unpacked।

### लोकल कमांड्स

```bash
npm run setup
npm run start
npm run verify
npm run doctor
```

यदि `.env` में एक असली क्लाउड LLM की है, तो XMem उस प्रोवाइडर का उपयोग करता है और FastEmbed के साथ एम्बेडिंग्स को लोकल रखता है। यदि कोई क्लाउड की कॉन्फ़िगर नहीं की गई है, तो XMem लोकल Ollama को फॉलबैक करता है और सेटअप के दौरान आवश्यक लोकल मॉडल्स खींचता है।

### कॉन्टेक्स्ट पोर्टेबिलिटी

```bash
npm run context:export
npm run context:import -- --file ./exports/xmem-context.json
npm run context:sync -- --file ./exports/xmem-context.json --server https://api.xmem.in --api-key <key>
```

`context:export` एक लोकल कॉन्टेक्स्ट बंडल लिखता है जिसे बाद में इंपोर्ट किया जा सकता है या XMem सर्वर को सिंक किया जा सकता है।

### एक रिपोज़िटरी को इंडेक्स करें

```bash
python -m src.scanner.runner \
  --org your-org \
  --repo your-repo \
  --url https://github.com/your-org/your-repo.git \
  --enrich
```

> [!टिप]
> पूरी तरह लोकल सेटअप के लिए कोई क्लाउड डिपेंडेंसी नहीं:
> ```ini
> FALLBACK_ORDER='["ollama"]'
> EMBEDDING_PROVIDER=ollama
> VECTOR_STORE_PROVIDER=pgvector
> ```
> फिर लोकल एक्सट्रास इंस्टॉल करें: `pip install -e ".[local]"`

## कॉन्फ़िगरेशन

पूरी कॉन्फ़िगरेशन जानकारी के लिए [docs/configuration.md](https://github.com/XortexAI/XMem/blob/main/docs/configuration.md) देखें।

## योगदान

हम योगदान का स्वागत करते हैं! कृपया [CONTRIBUTING.md](https://github.com/XortexAI/XMem/blob/main/CONTRIBUTING.md) पढ़ें पहले।

## लाइसेंस

यह प्रोजेक्ट BSD-3-Clause लाइसेंस के तहत लाइसेंस प्राप्त है। विवरण के लिए [LICENSE](LICENSE) देखें।

## संपर्क

प्रश्न या सहायता के लिए, कृपया [GitHub Issues](https://github.com/XortexAI/XMem/issues) पर एक समस्या खोलें।
