"""
chathelper.py - utils to help out parsing the chat

Memoir+ a persona extension for Text Gen Web UI. 
MIT License

Copyright (c) 2024 brucepro

"""
from extensions.Memoir.persona.persona import Persona
from extensions.Memoir.commands.urlhandler import UrlHandler
import re
from sqlite3 import connect
import pathlib
import validators


class ChatHelper():
    def __init__(self):
        pass

    def process_string(self, input_string):
        pattern = r'\[([^\[\]]+)\]'
        emotion_output = {}
        commands_in_string = re.findall(pattern, input_string, re.IGNORECASE)
        print("Processing commands:" + str(commands_in_string))

    def safer_string(self, input_string):
        # output_string = input_string.Replace("'","''");
        cleaned_string = re.sub(r'[^a-zA-Z0-9\s]+', '', input_string)
        return cleaned_string

    def remove_dtime(self, input_string):
        pattern = r"\[DTime=.*?\]"
        new_str = re.sub(pattern, "", input_string)
        return new_str

    def check_if_narration(input_string):
        # pattern check if it is narration.
        # set input name for narrator.
        if len(input_string) > 0:
            if input_string[0] == "*" and input_string[-1] == "*":
                return True
        return False
