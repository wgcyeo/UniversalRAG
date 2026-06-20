ROUTER_PROMPT = """Classify the following query into one or more categories from: [no, paragraph, document, table, image, clip, video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- no: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- table: The query requires information that is best represented in a tabular format, often involving comparisons or structured data.
- image: The query focuses on visual aspects like appearances, structures, or spatial relationships.
- clip: The query targets a short, specific moment or event within a video, without needing full context.
- video: The query requires understanding dynamic events, motion, or sequences over time in a video.

For cross-modality queries that require multiple types of information, combine categories with '+' (e.g., paragraph+image, document+video).

Examples:
1. "What is the capital of France?" -> no
2. "What is the birth date of Alan Turing?" -> paragraph
3. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" -> document
4. "Among the recipients of the Turing Award, who had the earliest birth year?" -> table
5. "Describe the appearance of a blue whale." -> image
6. "Describe the moment Messi scored his goal in the 2022 World Cup final." -> clip
7. "Explain how Messi scored his goal in the 2022 World Cup final." -> video
8. "Solve 12 x 8." -> no
9. "Who played a key role in the development of the iPhone?" -> paragraph
10. "Which Harvard University graduate played a key role in the development of the iPhone?" -> document
11. "What is the cheapest iPhone model available in 2023?" -> table
12. "Describe the structure of the Eiffel Tower." -> image
13. "Describe the moment Darth Vader reveals he is Luke's father in Star Wars." -> clip
14. "Analyze the sequence of events leading to the fall of the Empire in Star Wars." -> video
15. "Describe the visual appearance and habitat of the blue whale." -> paragraph+image
16. "Compare the architectural features shown in Gothic and Renaissance cathedrals." -> image+table
17. "Describe the moment of the moon landing and explain the mission details." -> paragraph+clip

Classify the following query: {query}
Provide only the category or categories combined with '+'.
"""
