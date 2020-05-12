from translate import Translator
translator = Translator(to_lang="ru", from_lang="en")
translation = translator.translate("Hello, world")
print(translation)

