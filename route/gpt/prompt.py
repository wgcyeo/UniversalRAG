ROUTER_PROMPT = """
Classify the following query into one of six categories: [No, Paragraph, Document, Image, Clip, Video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- No: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- Paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- Image: The query focuses on visual aspects like appearances, structures, or spatial relationships.
- Clip: The query targets a short, specific moment or event within a video, without needing full context.
- Video: The query requires understanding dynamic events, motion, or sequences over time in a video.

Examples:
1. "What is the capital of France?" → No
2. "What is the birth date of Alan Turing?" → Paragraph
3. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → Document
4. "Describe the appearance of a blue whale." → Image
5. "Describe the moment Messi scored his goal in the 2022 World Cup final." → Clip
6. "Explain how Messi scored his goal in the 2022 World Cup final." → Video
7. "Solve 12 × 8." → No
8. "Who played a key role in the development of the iPhone?" → Paragraph
9. "Which Harvard University graduate played a key role in the development of the iPhone?" → Document
10. "Describe the structure of the Eiffel Tower." → Image
11. "Describe the moment Darth Vader reveals he is Luke's father in Star Wars." → Clip
12. "Analyze the sequence of events leading to the fall of the Empire in Star Wars." → Video

Classify the following query: {query}
Provide only the category.
"""