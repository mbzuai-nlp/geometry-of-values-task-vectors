You are a helpful assistant assisting me in creating a diverse and unique set of realistic ethical dilemmas.  
 
An **ethical dilemma** is a situation where a person faces a conflict between two or more moral principles or values, each leading to a different course of action. These dilemmas arise when competing interests exist, and no solution fully satisfies all ethical considerations. 
 
### Example 
A common ethical dilemma might involve choosing between: 
- Telling the truth about a situation that could harm someone (aligning with the value of honesty), and 
- Protecting someone's feelings by withholding the truth (aligning with the value of compassion).

Both choices could be justified by different moral values, creating a difficult decision-making scenario. 
 
### Scenario 
For this task, you will focus on the {SCENARIO_NAME} context, which includes the following sub-scenarios: 
{SCENARIO_SUBLIST}
 
Your goal is to create **20 unique ethical dilemmas** in these contexts. Each dilemma should explore a conflict between the following two values: 
1. **{VALUE_1}** 
2. **{VALUE_2}**
 
### Output Format 
Return the dilemmas as a JSON object, where each dilemma has the following fields: 
- `"index"`: The serial number of the dilemma, 1,2,3,...
- `"story"`: A 4-5 line narrative describing the situation or setting where the dilemma arises. This should be descriptive enough to make sense of what the situation is in detail. This should not be less than 4 sentences.
- `"question"`: The specific question or decision the character/entity is faced with, phrased as: *Should they do A or B?* 
- `"options"`: A list containing the two possible actions: 
  - **A**: The action that prioritizes **{VALUE_1}**.
  - **B**: The action that prioritizes **{VALUE_2}**.
