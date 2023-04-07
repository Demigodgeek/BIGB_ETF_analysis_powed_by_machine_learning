def questionnaire():
    print("Is BIGB a wise choice for you?\n"
      "The BIG Bank ETF (BIGB) is an exchange-traded fund launched by Roundhill on NASDAQ on Mar 21, 2023\n"
      "It provides a concentrated and cost-efficient exposure to the six largest U.S. banks:\n"
      "Bank of America, Citigroup, Goldman Sachs, JPMorgan Chase, Morgan Stanley, and Wells Fargo.\n"
      "It is equally weighted, re-balances quarterly, and reconstitutes on an annual basis.\n"
      "The expense ratio of BIGB is 0.29%.\n"
      "Here are some questions to help us provide personalized recommendations and advice for you.\n\n"
)
    print("")

    # Get user's investment horizon
    while True:
        print("What is your investment horizon?")
        print("1. 1 month")
        print("2. 3 months")
        print("3. 6 months")
        print("4. 12 months")
        investment_horizon_choice = input("Enter your choice (1-4): ")
        if investment_horizon_choice in ['1', '2', '3', '4']:
            investment_horizon = int(investment_horizon_choice)
            break
        else:
            print("Invalid choice. Please try again.")
    
    # Get user's initial capital
    print("Please enter your initial capital (in USD), which must be larger than 10,000.")
    while True:
        initial_capital = input("Initial capital: ")
        if initial_capital.isnumeric() and int(initial_capital) > 10000:
            initial_capital = int(initial_capital)
            break
        else:
            print("Invalid input. Please enter a positive integer larger than 10,000.")
        
    
    # Get user's risk tolerance level
    while True:
        print("What is your risk tolerance level?")
        print("1. Conservative")
        print("2. Moderately Conservative")
        print("3. Moderate")
        print("4. Moderately Aggressive")
        print("5. Aggressive")
        risk_tolerance_choice = input("Enter your choice (1-5): ")
        if risk_tolerance_choice in ['1', '2', '3', '4', '5']:
            risk_tolerance_level = int(risk_tolerance_choice)
            break
        else:
            print("Invalid choice. Please try again.")

    return investment_horizon, initial_capital, risk_tolerance_level
