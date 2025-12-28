def inject_css():
    return '''
    <style>
    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }
    </style>
    '''