def questionnaire():
    
    print("")
    print("")
    print("")
    print("Is BIGB a wise choice for you?")
    print("")
    print("Here are some questions to help us provide personalized recommendations and advice for you.")
    # Get user's holding period
    print("Please enter the number of months you want to hold this BIGB, which must be an integer between 1 and 36 (inclusive)")
    
    while True:
        holding_period_choice = input("Holding Period: ")
        if holding_period_choice.isdigit() and 1 <= int(holding_period_choice) <= 36:
            holding_period = int(holding_period_choice)
            break
        else:
            print("Invalid choice. Please enter an integer between 1 and 36 (inclusive).")


    # Get user's initial capital
    print("Please enter your initial capital (in USD), which must be at least 10,000.")
    while True:
        initial_capital_choice = input("Initial Capital: ")
        if initial_capital_choice.isdigit() and int(initial_capital_choice) >= 10000:
            initial_capital = int(initial_capital_choice)
            break
        else:
            print("Invalid input. Please enter an integer greater than or equal to 10,000.")

    return holding_period, initial_capital