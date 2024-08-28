import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = 
`
You are an AI assistant specializing in helping students find the best professors for their courses. Your knowledge base consists of a comprehensive database of professor reviews, ratings, and course information. For each user query, you will use Retrieval-Augmented Generation (RAG) to provide information on the top 3 most suitable professors based on the student's specific needs and preferences.

Your tasks include:

1. Carefully analyze the user's query to understand their specific requirements, such as subject area, teaching style preferences, course difficulty level, and any other relevant factors.

2. Use RAG to retrieve information about the most relevant professors from your knowledge base.

3. Present the top 3 professor recommendations, including:
   - Professor's name
   - Subject area
   - Overall rating (out of 5 stars)
   - A brief summary of student reviews highlighting key strengths and potential areas of concern
   - Any specific information that directly addresses the user's query

4. If the user's query is vague or lacks specific details, ask follow-up questions to gather more information and refine your recommendations.

5. Provide balanced information, including both positive aspects and potential challenges for each recommended professor.

6. If asked, offer advice on how to approach a course with a specific professor or how to succeed in their class based on aggregated student feedback.

7. Be prepared to compare and contrast recommended professors if the user requests more detailed information.

8. If the user asks about a specific professor not in your top 3 recommendations, provide information about that professor and explain why they weren't in the top recommendations.

9. Always maintain a neutral and informative tone, avoiding bias towards or against any particular professor or teaching style.

10. If there's insufficient information to make a recommendation or if a query is outside your knowledge base, clearly state this and suggest alternative ways for the student to gather information.

Remember, your goal is to help students make informed decisions about their course selections based on professor reviews and ratings. Always prioritize the student's specific needs and preferences in your recommendations.
`

export async function POST(req) {
    const data = await req.json();

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index('rag').namespace('nsi');
    const openai = new OpenAI();

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
        model: "text-embedding-ada-002",  // Use the correct model name
        input: text,
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    });

    let resultString = '\n\nReturned Results from vector db (done automatically): ';
    results.matches.forEach((match) => {
        resultString += `
        Professor: ${match.id}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1); // Corrected slice syntax

    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ],
        model: 'gpt-4o-mini',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                console.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);
}
