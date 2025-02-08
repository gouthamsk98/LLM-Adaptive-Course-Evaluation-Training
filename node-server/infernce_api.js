import express from "express";
import * as hub from "langchain/hub";
import { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "langchain/output_parsers";

const app = express();
const port = 3002;

// Middleware to parse JSON requests
app.use(express.json());

async function runEvaluation(question, correct_answer, student_answer) {
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    result: "Pass or Fail",
    explanation: "Explanation of the result",
    score: "Numerical score from 0 to 100",
  });

  const formatInstructions = outputParser.getFormatInstructions();
  try {
    const prompt = await hub.pull("gouthamsk/rag-port11"); // Or your specific prompt
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const formattedPrompt = await prompt.format({
      context: " ",
      question: question + formatInstructions,
      correct_answer,
      student_answer,
    });

    const response = await llm.invoke(formattedPrompt);
    const parsedOutput = await outputParser.parse(response.content);

    return parsedOutput;
  } catch (error) {
    console.error("Error during RAG process:", error);
    throw new Error("An error occurred while processing your request.");
  }
}
async function runHint(question, student_answer, hint_level) {
  console.log("hint entered");
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    hintMessage: "A hint based on the question and user answer.",
    hintLineNumber: "Line number of the hint in the code",
  });
  const formatInstructions = outputParser.getFormatInstructions();
  try {
    const prompt = await hub.pull("gouthamsk/port11-hint"); // Or your specific prompt
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const formattedPrompt = await prompt.format({
      question: question + formatInstructions,
      user_answer: student_answer,
      hint_level: hint_level,
    });

    const response = await llm.invoke(formattedPrompt);
    const parsedOutput = await outputParser.parse(response.content);

    return parsedOutput;
  } catch (error) {
    console.error("Error during RAG process:", error);
    throw new Error("An error occurred while processing your request.");
  }
}
app.post("/evaluate", async (req, res) => {
  const { question, ref_answer, userCode, queryType, hint_level } = req.body;

  if (!question || !userCode || !queryType) {
    return res.status(400).json({
      error: "Missing required fields: question, correct_answer and queryType",
    });
  }

  try {
    if (queryType === "hint") {
      const result = await runHint(question, userCode, hint_level);
      result.hintLineNumber = parseInt(result.hintLineNumber);
      res.json(result);
    } else {
      const result = await runEvaluation(question, ref_answer, userCode);
      console.log(result);
      result.score = parseInt(result.score);
      if (result.result === "Fail") {
        result.score = 0;
      }

      res.json(result);
    }
  } catch (error) {
    res.status(500).json({
      error: "An error occurred while processing your request.",
    });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
