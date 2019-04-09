
def smooth_user_preference(x):
    return math.log(1+x, 2)

"encode event type"
def get_event_strength(x, duration = None):
    if x == '6' or x == 'ViewDuration':
        if duration >= 3000 and duration <= 10000:
            return 2 + (duration - 3000) / (10000 - 3000)  # the bonus value increase following by linear 
        if duration > 10000:
            return 3
        else:
            return "nothing"
    else:
        return event_type_strength.get(x, "nothing")

def from_number_to_array(number):
    result = []
    for index in range(number):
        result.append(index)
    return result

def fake_created_date(contentId):
    if int(contentId % 4) == 0:
        return (datetime.datetime.now() + datetime.timedelta(days = -4) ).strftime('%Y-%m-%d %H:%M')
    if int(contentId % 4) == 1:
        return (datetime.datetime.now() + datetime.timedelta(days = -3) ).strftime('%Y-%m-%d %H:%M') 
    if int(contentId % 4) == 2:
        return (datetime.datetime.now() + datetime.timedelta(days = -2) ).strftime('%Y-%m-%d %H:%M')
    else:
        return (datetime.datetime.now() + datetime.timedelta(days = -1) ).strftime('%Y-%m-%d %H:%M')

