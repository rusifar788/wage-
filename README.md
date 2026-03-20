# Wage Highlighter - Manual Review Mode

This version adds analyst review before highlighting.

## Run

```bat
py -3.12 -m pip install -r requirements.txt
py -3.12 app.py
```

Then open `http://127.0.0.1:5000`

## How it works

1. Upload wage register PDF and bank statement PDF.
2. App extracts employee names and wage amounts.
3. For each employee, app shows similar bank narration candidates.
4. Click **Yes – Highlight selected candidate** or **No – Skip this employee**.
5. At the end, download highlighted PDF and CSV report.

## Notes

- Useful when the payroll name is split across multiple lines.
- Useful when the bank credited salary in split entries or with small differences.
- This version highlights approved candidate lines and nearby amount lines.
